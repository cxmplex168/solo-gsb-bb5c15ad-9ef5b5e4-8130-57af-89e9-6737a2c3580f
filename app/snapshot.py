"""多权威节点 RRset 快照一致性检查（不落库）。

输入为多个权威节点各自上报的区域 RRset 快照，每个 RRset 含
``owner`` / ``type`` / ``class`` / ``ttl`` / ``rdata``。处理流程：

1. **规范化**
   - ``owner``：去首尾空白、转小写、去末尾根点；``@`` 视为区域顶点
     （替换为路径中的区域名），根节点 ``.`` 保留；
   - ``type`` / ``class``：去首尾空白并转大写（格式校验在请求模型中完成）；
   - ``rdata``：逐条去首尾空白后**去重并排序**，即 RRset 内记录无序、
     以集合比较；RDATA 文本本身大小写敏感、原样保留；
   - 以 ``(owner, type, class)`` 作为 RRset 键。
2. **逐 RRset 多数共识**：每个节点对同一键至多贡献一个变体
   （签名为 ``(ttl, 排序后的 rdata 元组)``，TTL 不同即视为不同变体）。
   取得支持节点最多的变体；支持数达到 ``min_agreement`` 即进入可发布的
   共识快照，否则记入冲突 RRset。最高支持数并列时按变体签名
   （rdata 序列、再 TTL）字典序升序决出确定结果。
3. **逐节点差异**：相对共识快照列出每个节点 ``missing``（共识有、本端缺）、
   ``extra``（本端有、未达共识）与 ``mismatched``（双方都有但变体不同）。

任一 RRset 未达门槛时抛 :class:`SnapshotConsensusError`（API 层转换为 409），
detail 中仍携带已达成共识的部分快照与逐节点差异；同一节点对同一 RRset 键
上报互相矛盾的多个变体时抛 :class:`SnapshotValidationError`（API 层转换为 422）。
"""
from __future__ import annotations

# SnapshotConsensusError.detail.reason：存在未达最小同意节点数的 RRset
ERR_INSUFFICIENT_CONSENSUS = "insufficient_consensus"

# SnapshotValidationError.code：同一节点快照内同一 RRset 键出现矛盾变体
ERR_DUPLICATE_RRSET = "duplicate_rrset_in_snapshot"


class SnapshotValidationError(Exception):
    """节点快照自身不合法（对应 HTTP 422）。

    :param conflicts: 矛盾 RRset 描述列表（节点名 + RRset 键）。
    """

    def __init__(self, message: str, conflicts: list[dict]) -> None:
        super().__init__(message)
        self.code = ERR_DUPLICATE_RRSET
        self.conflicts = conflicts


class SnapshotConsensusError(Exception):
    """存在达不到最小同意节点数的 RRset（对应 HTTP 409 Conflict）。

    :param detail: 完整的冲突响应体（reason/message/冲突 RRset/部分共识/逐节点差异）。
    """

    def __init__(self, message: str, detail: dict) -> None:
        super().__init__(message)
        self.detail = detail


def normalize_owner(owner: str, zone: str) -> str:
    """规范化 RRset owner 名：小写、去末尾根点；``@`` 替换为区域顶点。"""
    value = owner.strip().lower()
    if value == "@":
        value = zone.strip().lower()
    if value == ".":
        return "."
    return value.rstrip(".")


def _canonical_rrset(
    raw: tuple[str, str, str, int, list[str]], zone: str
) -> tuple[tuple[str, str, str], tuple[int, tuple[str, ...]], dict]:
    """把一条原始 RRset 转为 (键, 变体签名, 规范化记录)。

    键为 ``(owner, type, class)``；变体签名为 ``(ttl, 排序去重后的 rdata 元组)``。
    """
    owner, rrtype, rrclass, ttl, rdata = raw
    canonical_owner = normalize_owner(owner, zone)
    canonical_rdata = tuple(sorted({item.strip() for item in rdata}))
    key = (canonical_owner, rrtype, rrclass)
    signature = (ttl, canonical_rdata)
    record = {
        "owner": canonical_owner,
        "type": rrtype,
        "class": rrclass,
        "ttl": ttl,
        "rdata": list(canonical_rdata),
    }
    return key, signature, record


def _rrset_out(key: tuple[str, str, str], signature: tuple[int, tuple[str, ...]]) -> dict:
    """由键与变体签名构造规范化 RRset 输出。"""
    owner, rrtype, rrclass = key
    ttl, rdata = signature
    return {
        "owner": owner,
        "type": rrtype,
        "class": rrclass,
        "ttl": ttl,
        "rdata": list(rdata),
    }


def build_snapshot_consistency(
    zone: str,
    nodes: list[tuple[str, list[tuple[str, str, str, int, list[str]]]]],
    min_agreement: int,
) -> dict:
    """计算多权威节点 RRset 快照的多数共识；不写库。

    :param zone: 已校验的区域名（用于 ``@`` owner 归一化）；
    :param nodes: ``(节点名, [RRset...])`` 列表，节点顺序即请求顺序；
                  RRset 为 ``(owner, type, class, ttl, rdata)``；
    :param min_agreement: RRset 进入共识快照所需的最小同意节点数（>=1，
                          且不超过节点数，边界由请求模型保证）。
    :raises SnapshotValidationError: 同一节点对同一 RRset 键给出矛盾变体（422）；
    :raises SnapshotConsensusError: 存在未达门槛的 RRset（409）。
    """
    node_names = [name for name, _ in nodes]
    all_names = set(node_names)

    # 每节点规范化后的 {键: (签名, 记录)}；矛盾重复在此拦截
    node_records: list[tuple[str, dict[tuple, tuple]]] = []
    duplicate_conflicts: list[dict] = []
    for name, rrsets in nodes:
        normalized: dict[tuple, tuple] = {}
        for raw in rrsets:
            key, signature, record = _canonical_rrset(raw, zone)
            existing = normalized.get(key)
            if existing is not None:
                if existing[0] != signature:
                    owner, rrtype, rrclass = key
                    duplicate_conflicts.append(
                        {
                            "node": name,
                            "owner": owner,
                            "type": rrtype,
                            "class": rrclass,
                        }
                    )
                # 完全重复的规范化记录按一条处理
                continue
            normalized[key] = (signature, record)
        node_records.append((name, normalized))

    if duplicate_conflicts:
        duplicate_conflicts.sort(key=lambda item: (item["node"], item["owner"], item["type"], item["class"]))
        raise SnapshotValidationError(
            "同一权威节点的快照中对相同 (owner, type, class) 上报了不一致的多个 RRset",
            conflicts=duplicate_conflicts,
        )

    # 键 -> 变体签名 -> 支持该变体的节点名列表
    variants_by_key: dict[tuple, dict[tuple, list[str]]] = {}
    for name, normalized in node_records:
        for key, (signature, _record) in normalized.items():
            variants_by_key.setdefault(key, {}).setdefault(signature, []).append(name)

    consensus: dict[tuple, tuple] = {}  # 键 -> 获胜变体签名
    conflicts: list[dict] = []
    for key in sorted(variants_by_key):
        variants = variants_by_key[key]
        # 支持数多者胜；并列时按 (rdata 序列, ttl) 字典序升序，保证结果确定
        best_signature, supporters = min(
            variants.items(),
            key=lambda item: (-len(item[1]), item[0][1], item[0][0]),
        )
        if len(supporters) >= min_agreement:
            consensus[key] = best_signature
            continue

        present_names = {name for names in variants.values() for name in names}
        variant_list = [
            {
                "ttl": signature[0],
                "rdata": list(signature[1]),
                "nodes": sorted(names),
            }
            for signature, names in sorted(
                variants.items(),
                key=lambda item: (-len(item[1]), item[0][1], item[0][0]),
            )
        ]
        conflicts.append(
            {
                "owner": key[0],
                "type": key[1],
                "class": key[2],
                "best_agreement": max(len(names) for names in variants.values()),
                "variants": variant_list,
                "missing_nodes": sorted(all_names - present_names),
            }
        )

    consensus_snapshot: list[dict] = []
    for key in sorted(consensus):
        signature = consensus[key]
        agreed = set(variants_by_key[key][signature])
        consensus_snapshot.append(
            {
                **_rrset_out(key, signature),
                "agreed_nodes": sorted(agreed),
                "dissenting_nodes": sorted(all_names - agreed),
            }
        )

    # 逐节点差异（相对已达成共识的 RRset；409 时未达共识键计入 extra）
    node_diffs: list[dict] = []
    for name, normalized in node_records:
        missing: list[dict] = []
        extra: list[dict] = []
        mismatched: list[dict] = []
        for key in sorted(set(consensus) | set(normalized)):
            if key not in normalized:
                missing.append(_rrset_out(key, consensus[key]))
                continue
            signature, record = normalized[key]
            if key not in consensus:
                extra.append(record)
            elif signature != consensus[key]:
                mismatched.append(
                    {
                        "owner": key[0],
                        "type": key[1],
                        "class": key[2],
                        "consensus": _rrset_out(key, consensus[key]),
                        "node_rrset": record,
                    }
                )
        node_diffs.append(
            {
                "node": name,
                "missing": missing,
                "extra": extra,
                "mismatched": mismatched,
            }
        )

    if conflicts:
        detail = {
            "reason": ERR_INSUFFICIENT_CONSENSUS,
            "message": (
                f"有 {len(conflicts)} 个 RRset 在 {len(node_names)} 个权威节点中"
                f"未能达到最小同意节点数 {min_agreement}，无法整体发布"
            ),
            "zone": zone,
            "node_count": len(node_names),
            "min_agreement": min_agreement,
            "conflicts": conflicts,
            "consensus_snapshot": consensus_snapshot,
            "node_diffs": node_diffs,
        }
        raise SnapshotConsensusError(detail["message"], detail)

    return {
        "zone": zone,
        "node_count": len(node_names),
        "min_agreement": min_agreement,
        "consensus_snapshot": consensus_snapshot,
        "node_diffs": node_diffs,
        "conflicts": [],
        "note": (
            "共识快照依据请求中的多权威节点 RRset 快照即时计算，不写入数据库；"
            "owner/type/class 已规范化、RRset 内 rdata 按无序集合比较，"
            "每个 RRset 取得不少于最小同意节点数的变体方可发布"
        ),
    }
