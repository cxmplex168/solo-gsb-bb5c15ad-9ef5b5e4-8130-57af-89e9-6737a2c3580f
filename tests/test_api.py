"""API 与 RFC 1982 算术的测试。"""
from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.rfc1982 import SERIAL_HALF, MAX_SERIAL, SERIAL_SPACE, classify
from app.suggest import daily_base, suggest_next_serial, SuggestionError

TODAY = date(2026, 9, 18)
BASE = 2026091800  # YYYYMMDD00


def _utc_today_base() -> int:
    """服务器 UTC 当日的 YYYYMMDD00。"""
    return daily_base(datetime.now(timezone.utc).date())


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


# ---------- RFC 1982 分类 ----------

@pytest.mark.parametrize(
    "prev,cur,expected",
    [
        (1, 2, "advance"),
        (5, 5, "equal"),
        (2, 1, "regression"),
        (0, SERIAL_HALF, "incomparable"),
        (0, SERIAL_HALF - 1, "advance"),
        (0, SERIAL_HALF + 1, "regression"),
        (MAX_SERIAL, 0, "advance"),          # 回绕后仍算前进
        (MAX_SERIAL - 9, 10, "advance"),     # 回绕前进 20
        (10, MAX_SERIAL - 9, "regression"),  # 反向回绕即倒退
    ],
)
def test_classify(prev, cur, expected):
    assert classify(prev, cur) == expected


# ---------- 审计接口 ----------

def test_create_and_get_audit(client):
    serials = [BASE, BASE + 1, BASE + 1, BASE - 5]
    resp = client.post("/api/audits", json={"zone": "example.com", "serials": serials})
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == 1
    assert body["zone"] == "example.com"
    assert body["serials"] == serials

    relations = [item["relation"] for item in body["items"]]
    assert relations == ["initial", "advance", "equal", "regression"]
    risks = [item["slave_reject_risk"] for item in body["items"]]
    assert risks == [False, False, False, True]
    assert body["summary"]["risk_indexes"] == [3]
    assert body["summary"]["regression"] == 1

    got = client.get(f"/api/audits/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body


def test_incomparable_flagged_as_risk(client):
    serials = [0, SERIAL_HALF]
    resp = client.post("/api/audits", json={"zone": "example.com", "serials": serials})
    assert resp.status_code == 201
    item = resp.json()["items"][1]
    assert item["relation"] == "incomparable"
    assert item["slave_reject_risk"] is True
    assert item["delta"] is None


def test_wraparound_sequence(client):
    serials = [MAX_SERIAL - 1, MAX_SERIAL, 0, 1]
    resp = client.post("/api/audits", json={"zone": "example.com.", "serials": serials})
    assert resp.status_code == 201
    relations = [i["relation"] for i in resp.json()["items"]]
    assert relations == ["initial", "advance", "advance", "advance"]


def test_get_missing_audit_404(client):
    assert client.get("/api/audits/999").status_code == 404


# ---------- 修复预演 ----------

def test_repair_plan_keeps_and_replaces(client):
    serials = [BASE, BASE + 1, BASE + 1, BASE - 5]
    created = client.post("/api/audits", json={"zone": "example.com", "serials": serials})
    audit_id = created.json()["id"]

    resp = client.get(f"/api/audits/{audit_id}/repair-plan")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == audit_id
    assert body["zone"] == "example.com"

    items = body["items"]
    assert [(i["index"], i["original"], i["effective"], i["action"]) for i in items] == [
        (0, BASE, BASE, "keep"),        # 首项不变
        (1, BASE + 1, BASE + 1, "keep"),  # 有效前进
        (2, BASE + 1, BASE + 2, "replace"),  # 相等 → 上一修复值 +1
        (3, BASE - 5, BASE + 3, "replace"),  # 相对修复后值倒退 → +1
    ]
    assert body["summary"] == {"total": 4, "replaced": 2}
    # 替换原因逐项返回
    assert "相等" in items[2]["reason"]
    assert "倒退" in items[3]["reason"]
    # 修复后序列相邻项均为有效前进
    effectives = [i["effective"] for i in items]
    assert all(classify(a, b) == "advance" for a, b in zip(effectives, effectives[1:]))


def test_repair_plan_wraparound(client):
    # 0 相对 MAX_SERIAL 是回绕前进（保留）；MAX_SERIAL 相对 0 是倒退（替换为 1）
    serials = [MAX_SERIAL, 0, MAX_SERIAL]
    audit_id = client.post(
        "/api/audits", json={"zone": "example.com", "serials": serials}
    ).json()["id"]

    body = client.get(f"/api/audits/{audit_id}/repair-plan").json()
    assert [(i["original"], i["effective"], i["action"]) for i in body["items"]] == [
        (MAX_SERIAL, MAX_SERIAL, "keep"),
        (0, 0, "keep"),
        (MAX_SERIAL, 1, "replace"),
    ]
    assert body["summary"]["replaced"] == 1


def test_repair_plan_replace_wraps_to_zero(client):
    # 上一修复后值为 2^32-1 且原值相等时，替换值按 32 位无符号回绕为 0
    serials = [MAX_SERIAL, MAX_SERIAL]
    audit_id = client.post(
        "/api/audits", json={"zone": "example.com", "serials": serials}
    ).json()["id"]

    items = client.get(f"/api/audits/{audit_id}/repair-plan").json()["items"]
    assert items[1]["effective"] == 0
    assert items[1]["action"] == "replace"
    assert classify(items[0]["effective"], items[1]["effective"]) == "advance"


def test_repair_plan_incomparable_replaced(client):
    serials = [0, SERIAL_HALF]
    audit_id = client.post(
        "/api/audits", json={"zone": "example.com", "serials": serials}
    ).json()["id"]

    items = client.get(f"/api/audits/{audit_id}/repair-plan").json()["items"]
    assert items[1]["action"] == "replace"
    assert items[1]["effective"] == 1
    assert "不可比较" in items[1]["reason"]


def test_repair_plan_all_advance_kept(client):
    serials = [MAX_SERIAL - 1, MAX_SERIAL, 0, 1]
    audit_id = client.post(
        "/api/audits", json={"zone": "example.com.", "serials": serials}
    ).json()["id"]

    body = client.get(f"/api/audits/{audit_id}/repair-plan").json()
    assert all(i["action"] == "keep" for i in body["items"])
    assert [i["effective"] for i in body["items"]] == serials
    assert body["summary"] == {"total": 4, "replaced": 0}


def test_repair_plan_single_serial(client):
    audit_id = client.post(
        "/api/audits", json={"zone": "example.com", "serials": [42]}
    ).json()["id"]

    body = client.get(f"/api/audits/{audit_id}/repair-plan").json()
    assert body["items"] == [
        {"index": 0, "original": 42, "effective": 42, "action": "keep",
         "reason": body["items"][0]["reason"]}
    ]
    assert body["summary"] == {"total": 1, "replaced": 0}


def test_repair_plan_not_persisted(client):
    serials = [BASE, BASE - 5]
    audit_id = client.post(
        "/api/audits", json={"zone": "example.com", "serials": serials}
    ).json()["id"]

    client.get(f"/api/audits/{audit_id}/repair-plan")
    # 原审计记录未被修改
    stored = client.get(f"/api/audits/{audit_id}").json()
    assert stored["serials"] == serials


def test_repair_plan_missing_audit_404(client):
    assert client.get("/api/audits/999/repair-plan").status_code == 404


# ---------- 下一序列号建议 ----------


def test_list_audits(client):
    client.post("/api/audits", json={"zone": "a.example", "serials": [1, 2]})
    client.post("/api/audits", json={"zone": "b.example", "serials": [2, 1]})
    resp = client.get("/api/audits")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert rows[0]["id"] == 2  # 倒序
    assert rows[0]["risk_count"] == 1
    assert rows[1]["risk_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"zone": "example.com", "serials": []},                    # 空列表
        {"zone": "example.com", "serials": [-1]},                  # 越界
        {"zone": "example.com", "serials": [MAX_SERIAL + 1]},      # 越界
        {"zone": "bad_zone!", "serials": [1]},                     # 非法区域名
        {"zone": "", "serials": [1]},                              # 空区域名
        {"zone": "example.com"},                                   # 缺字段
    ],
)
def test_invalid_payload_422(client, payload):
    assert client.post("/api/audits", json=payload).status_code == 422


# ---------- 下一序列号建议 ----------

def test_suggest_when_behind_base():
    assert suggest_next_serial(2026010100, TODAY) == BASE
    assert suggest_next_serial(0, TODAY) == BASE
    # 回绕场景：接近 2^32 上限时，今日基准仍是有效前进
    assert suggest_next_serial(MAX_SERIAL, TODAY) == BASE


def test_suggest_within_today():
    assert suggest_next_serial(BASE, TODAY) == BASE + 1
    assert suggest_next_serial(BASE + 50, TODAY) == BASE + 51
    assert suggest_next_serial(BASE + 98, TODAY) == BASE + 99


def test_suggest_exhausted():
    with pytest.raises(SuggestionError):
        suggest_next_serial(BASE + 99, TODAY)


def test_suggest_ahead_of_today():
    with pytest.raises(SuggestionError):
        suggest_next_serial(BASE + 100, TODAY)
    with pytest.raises(SuggestionError):
        suggest_next_serial(2099010100, TODAY)  # 未来日期


def test_suggest_half_interval_ambiguity():
    ambiguous = (BASE + SERIAL_HALF) % (MAX_SERIAL + 1)
    with pytest.raises(SuggestionError):
        suggest_next_serial(ambiguous, TODAY)


def test_next_serial_endpoint(client):
    resp = client.get("/api/next-serial", params={"current": BASE + 7, "today": "2026-09-18"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["next"] == BASE + 8
    assert body["date"] == "20260918"
    # 建议值确为有效前进
    assert classify(body["current"], body["next"]) == "advance"


def test_next_serial_endpoint_409(client):
    resp = client.get("/api/next-serial", params={"current": BASE + 99, "today": "2026-09-18"})
    assert resp.status_code == 409
    ambiguous = (BASE + SERIAL_HALF) % (MAX_SERIAL + 1)
    resp = client.get("/api/next-serial", params={"current": ambiguous, "today": "2026-09-18"})
    assert resp.status_code == 409


def test_next_serial_default_date(client):
    resp = client.get("/api/next-serial", params={"current": 1})
    assert resp.status_code == 200
    assert classify(1, resp.json()["next"]) == "advance"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# ---------- SOA serial 持久化预留 ----------


def test_reserve_first_and_next(client):
    base = _utc_today_base()
    resp = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": base - 1, "request_id": "req-1"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["serial"] == base  # 落后当日基准 → 分配 YYYYMMDD00
    assert body["current"] == base - 1
    assert body["zone"] == "example.com"
    assert body["request_id"] == "req-1"
    assert body["replayed"] is False
    assert body["date"] == datetime.now(timezone.utc).strftime("%Y%m%d")

    # 第二次必须以上一次分配值为 current
    resp2 = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": body["serial"], "request_id": "req-2"},
    )
    assert resp2.status_code == 201
    assert resp2.json()["serial"] == base + 1


def test_reserve_idempotent_replay_returns_first_result(client):
    base = _utc_today_base()
    payload = {"current": base, "request_id": "dup-req"}
    first = client.post("/api/zones/example.com/serial-reservations", json=payload)
    assert first.status_code == 201
    assert first.json()["replayed"] is False

    # 同 zone + 同 request_id 重试：200 + 首次结果，不再次推进
    replay = client.post("/api/zones/example.com/serial-reservations", json=payload)
    assert replay.status_code == 200
    replay_body = replay.json()
    assert replay_body == {**first.json(), "replayed": True}

    # 即使重试时携带不同 current，仍返回首次结果
    replay2 = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": base + 99, "request_id": "dup-req"},
    )
    assert replay2.status_code == 200
    assert replay2.json()["serial"] == first.json()["serial"]
    assert replay2.json()["current"] == base

    # 预留记录只有一条
    rows = client.get("/api/zones/example.com/serial-reservations").json()
    assert [r["serial"] for r in rows] == [first.json()["serial"]]


def test_reserve_stale_current_409_with_latest(client):
    base = _utc_today_base()
    first = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": base, "request_id": "a"},
    ).json()
    latest = first["serial"]

    # 用过期 current 请求（新 request_id）→ 409 并带回最新值
    resp = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": base, "request_id": "b"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "stale_current"
    assert detail["latest_serial"] == latest

    # 按最新值重试可成功
    ok = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": latest, "request_id": "b"},
    )
    assert ok.status_code == 201
    assert ok.json()["serial"] == latest + 1


def test_reserve_nn_exhausted_409(client):
    base = _utc_today_base()
    resp = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": base + 99, "request_id": "exhaust"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "nn_exhausted"
    assert detail["latest_serial"] is None


def test_reserve_half_interval_409(client):
    base = _utc_today_base()
    ambiguous = (base + SERIAL_HALF) % SERIAL_SPACE
    resp = client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": ambiguous, "request_id": "amb"},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "half_interval_ambiguous"


def test_reserve_zones_are_independent(client):
    base = _utc_today_base()
    r1 = client.post(
        "/api/zones/a.example/serial-reservations",
        json={"current": base, "request_id": "x"},
    )
    r2 = client.post(
        "/api/zones/b.example/serial-reservations",
        json={"current": base, "request_id": "x"},
    )
    assert r1.status_code == r2.status_code == 201
    # 不同区域互不影响 current 校验，且同一 request_id 可各自使用
    assert r1.json()["serial"] == r2.json()["serial"] == base + 1

    rows = client.get("/api/zones/a.example/serial-reservations").json()
    assert [r["zone"] for r in rows] == ["a.example"]


def test_reserve_list_order_and_pagination(client):
    base = _utc_today_base()
    current = base
    for i in range(3):
        body = client.post(
            "/api/zones/example.com/serial-reservations",
            json={"current": current, "request_id": f"r{i}"},
        ).json()
        current = body["serial"]

    rows = client.get("/api/zones/example.com/serial-reservations").json()
    assert [r["serial"] for r in rows] == [base + 1, base + 2, base + 3]
    page = client.get(
        "/api/zones/example.com/serial-reservations", params={"limit": 2, "offset": 1}
    ).json()
    assert [r["serial"] for r in page] == [base + 2, base + 3]
    # 未知区域返回空列表
    assert client.get("/api/zones/nonexistent.example/serial-reservations").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"request_id": "r1"},                          # 缺 current
        {"current": MAX_SERIAL + 1, "request_id": "r1"},  # current 越界
        {"current": BASE, "request_id": 21},           # request_id 类型错误
    ],
)
def test_reserve_invalid_body_422(client, payload):
    resp = client.post(
        "/api/zones/example.com/serial-reservations", json=payload
    )
    assert resp.status_code == 422


def test_reserve_validation_errors(client):
    # current 越界
    assert client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": -1, "request_id": "r"},
    ).status_code == 422
    assert client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": MAX_SERIAL + 1, "request_id": "r"},
    ).status_code == 422
    # request_id 缺失 / 空白
    assert client.post(
        "/api/zones/example.com/serial-reservations", json={"current": BASE}
    ).status_code == 422
    assert client.post(
        "/api/zones/example.com/serial-reservations",
        json={"current": BASE, "request_id": "   "},
    ).status_code == 422
    # 路径中区域名非法
    assert client.post(
        "/api/zones/bad_zone!/serial-reservations",
        json={"current": BASE, "request_id": "r"},
    ).status_code == 400


def test_reserve_persisted_across_app_instances(tmp_path):
    db_path = str(tmp_path / "persist.db")
    base = _utc_today_base()
    with TestClient(create_app(db_path=db_path)) as c:
        body = c.post(
            "/api/zones/example.com/serial-reservations",
            json={"current": base, "request_id": "persist-1"},
        ).json()
        assert body["serial"] == base + 1

    # 重新打开同一数据库：记录仍在，且 current 校验以已预留最新值为准
    with TestClient(create_app(db_path=db_path)) as c:
        rows = c.get("/api/zones/example.com/serial-reservations").json()
        assert [r["serial"] for r in rows] == [base + 1]
        stale = c.post(
            "/api/zones/example.com/serial-reservations",
            json={"current": base, "request_id": "persist-2"},
        )
        assert stale.status_code == 409
        replay = c.post(
            "/api/zones/example.com/serial-reservations",
            json={"current": base, "request_id": "persist-1"},
        )
        assert replay.status_code == 200
        assert replay.json()["serial"] == base + 1


# ---------- 多权威节点共同发布规划 ----------


def test_release_plan_first_common_candidate(client):
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [
                {"name": "ns1", "serial": 2026091805},
                {"name": "ns2", "serial": 2026010100},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["zone"] == "example.com"
    assert body["date"] == "20260919"
    assert body["candidate"] == 2026091900  # 两个节点都落后 → 首个候选 YYYYMMDD00
    assert body["candidate_window"] == {"start": 2026091900, "end": 2026091999}
    assert [n["name"] for n in body["nodes"]] == ["ns1", "ns2"]
    assert all(n["relation"] == "advance" for n in body["nodes"])
    assert [n["forward_distance"] for n in body["nodes"]] == [95, 81800]
    # 每个候选确实是对各节点的有效前进
    assert all(classify(n["serial"], body["candidate"]) == "advance" for n in body["nodes"])


def test_release_plan_skips_non_advance_candidates(client):
    # 一个节点已在 base（00 相等），另一个已在 base+2（00..02 都不构成前进）
    # → 首个共同候选为 base+3
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [
                {"name": "a", "serial": 2026091900},
                {"name": "b", "serial": 2026091902},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate"] == 2026091903
    assert [n["forward_distance"] for n in body["nodes"]] == [3, 1]


def test_release_plan_wraparound_node(client):
    # 节点 serial 在 2^32 顶端：base 经回绕仍是有效前进
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [
                {"name": "near-max", "serial": MAX_SERIAL},
                {"name": "old", "serial": MAX_SERIAL - 10},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate"] == 2026091900
    assert [n["forward_distance"] for n in body["nodes"]] == [2026091901, 2026091911]


def test_release_plan_no_common_candidate_409_blockers(client):
    # A 的有效前进窗口为 00..69（在 70 处不可比较，71..99 倒退）；
    # B 已在 70（00..69 倒退、70 相等、71..99 前进）→ 无共同候选
    a = (2026091900 + 70 - SERIAL_HALF) % SERIAL_SPACE
    b = 2026091970
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [{"name": "a", "serial": a}, {"name": "b", "serial": b}],
        },
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "no_common_candidate"
    assert detail["date"] == "20260919"
    assert detail["candidate_window"] == {"start": 2026091900, "end": 2026091999}
    assert detail["checked"] == 100
    assert detail["blockers"] == [
        {"name": "a", "serial": a, "advance": 70, "equal": 0,
         "regression": 29, "incomparable": 1},
        {"name": "b", "serial": b, "advance": 29, "equal": 1,
         "regression": 70, "incomparable": 0},
    ]
    # 四类计数之和恒等于扫描候选数
    for blocker in detail["blockers"]:
        assert (
            blocker["advance"] + blocker["equal"]
            + blocker["regression"] + blocker["incomparable"]
        ) == 100


def test_release_plan_single_node_409_when_no_advance(client):
    # 唯一节点已在 YYYYMMDD99：区间内只有 equal/regression，无前进候选
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [{"name": "only", "serial": 2026091999}],
        },
    )
    assert resp.status_code == 409
    blocker = resp.json()["detail"]["blockers"][0]
    assert blocker["name"] == "only"
    assert blocker == {"name": "only", "serial": 2026091999,
                       "advance": 0, "equal": 1, "regression": 99, "incomparable": 0}


def test_release_plan_incomparable_appears_in_blocker_summary(client):
    # 节点在 +10 处与候选恰好相距 2^31（不可比较）；另一节点把可行窗口限制在 10 之后
    inc = (2026091900 + 10 - SERIAL_HALF) % SERIAL_SPACE  # 前进 00..09，10 不可比较，11..99 倒退
    late = 2026091909                                     # 00..08 倒退，09 相等，10..99 前进
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [{"name": "inc", "serial": inc}, {"name": "late", "serial": late}],
        },
    )
    assert resp.status_code == 409
    blockers = {b["name"]: b for b in resp.json()["detail"]["blockers"]}
    assert blockers["inc"]["incomparable"] == 1
    assert blockers["inc"]["advance"] == 10
    assert blockers["late"]["equal"] == 1
    assert blockers["late"]["advance"] == 90


def test_release_plan_date_out_of_range_409(client):
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={"today": "9999-12-31", "nodes": [{"name": "a", "serial": 1}]},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "date_out_of_range"
    assert detail["date"] == "99991231"
    assert detail["candidate_window"] is None
    assert detail["checked"] == 0
    assert detail["blockers"] == []


def test_release_plan_default_today(client):
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={"nodes": [{"name": "a", "serial": 1}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == datetime.now(timezone.utc).strftime("%Y%m%d")
    assert classify(1, body["candidate"]) == "advance"


def test_release_plan_not_persisted(client):
    resp = client.post(
        "/api/zones/example.com/release-plans",
        json={
            "today": "2026-09-19",
            "nodes": [
                {"name": "a", "serial": 2026091900},
                {"name": "b", "serial": 2026091902},
            ],
        },
    )
    assert resp.status_code == 200
    # 规划不写库：审计与预留均无记录
    assert client.get("/api/audits").json() == []
    assert client.get("/api/zones/example.com/serial-reservations").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"nodes": []},  # 空节点列表
        {"nodes": [{"name": "a", "serial": 1},
                   {"name": "a", "serial": 2}]},  # 节点名重复
        {"nodes": [{"name": "a", "serial": 1},
                   {"name": " a ", "serial": 2}]},  # 去除空白后重复
        {"nodes": [{"name": "   ", "serial": 1}]},  # 空白节点名
        {"nodes": [{"name": "a", "serial": -1}]},  # serial 下越界
        {"nodes": [{"name": "a", "serial": MAX_SERIAL + 1}]},  # serial 上越界
        {"nodes": [{"name": "a", "serial": 1.5}]},  # serial 非整数
        {"nodes": [{"serial": 1}]},  # 缺节点名
        "not-an-object",  # 请求体非对象
    ],
)
def test_release_plan_invalid_body_422(client, payload):
    resp = client.post(
        "/api/zones/example.com/release-plans", json=payload
    )
    assert resp.status_code == 422


def test_release_plan_invalid_zone_400(client):
    resp = client.post(
        "/api/zones/bad_zone!/release-plans",
        json={"nodes": [{"name": "a", "serial": 1}]},
    )
    assert resp.status_code == 400


# ---------- 区域传送收敛预演 ----------


def _simulation_payload(**overrides):
    payload = {
        "nodes": [
            {"name": "s", "serial": 100},
            {"name": "a", "serial": 90},
            {"name": "b", "serial": 80},
        ],
        "edges": [{"from": "s", "to": "a"}, {"from": "a", "to": "b"}],
        "source": "s",
        "publish_serial": 110,
        "max_rounds": 10,
    }
    payload.update(overrides)
    return payload


def test_transfer_simulation_linear_convergence(client):
    resp = client.post(
        "/api/zones/example.com/transfer-simulations", json=_simulation_payload()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["zone"] == "example.com"
    assert body["source"] == "s"
    assert body["publish_serial"] == 110
    assert body["max_rounds"] == 10
    assert body["converged"] is True
    assert body["sync_rounds_run"] == 2
    # 第 0 轮：仅源节点切换
    round0 = body["rounds"][0]
    assert round0["round"] == 0
    assert round0["changes"] == [
        {"node": "s", "previous_serial": 100, "serial": 110,
         "upstream": None, "forward_distance": 10}
    ]
    # 第 1 轮：a 直接拿到发布值；b 只从 a 的旧值(90)前进
    assert [(c["node"], c["previous_serial"], c["serial"], c["upstream"], c["forward_distance"])
            for c in body["rounds"][1]["changes"]] == [
        ("a", 90, 110, "s", 20),
        ("b", 80, 90, "a", 10),
    ]
    # 第 2 轮：b 从 a 的新值前进到发布值
    assert [(c["node"], c["serial"], c["upstream"])
            for c in body["rounds"][2]["changes"]] == [("b", 110, "a")]
    assert [(n["name"], n["serial"]) for n in body["snapshot"]] == [
        ("s", 110), ("a", 110), ("b", 110)
    ]
    assert body["unconverged"] == []


def test_transfer_simulation_round0_publish_equal_is_422(client):
    # 发布值与源节点当前值相等（非前进）→ 422
    resp = client.post(
        "/api/zones/example.com/transfer-simulations",
        json=_simulation_payload(publish_serial=100),
    )
    assert resp.status_code == 422


def test_transfer_simulation_publish_regression_is_422(client):
    # 发布值相对源节点倒退 → 422
    resp = client.post(
        "/api/zones/example.com/transfer-simulations",
        json=_simulation_payload(publish_serial=99),
    )
    assert resp.status_code == 422


def test_transfer_simulation_publish_incomparable_is_422(client):
    # 发布值与源节点相距恰好 2^31（不可比较）→ 422
    resp = client.post(
        "/api/zones/example.com/transfer-simulations",
        json=_simulation_payload(
            nodes=[{"name": "s", "serial": 0}], edges=[],
            publish_serial=SERIAL_HALF,
        ),
    )
    assert resp.status_code == 422


def test_transfer_simulation_min_forward_distance(client):
    # b 有两个上游：near=105（距离 5）与 far=120（距离 20），只取最近的 near
    payload = _simulation_payload(
        nodes=[
            {"name": "s", "serial": 100},
            {"name": "b", "serial": 100},
            {"name": "near", "serial": 105},
            {"name": "far", "serial": 120},
        ],
        edges=[{"from": "near", "to": "b"}, {"from": "far", "to": "b"}],
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    b_change = [c for c in body["rounds"][1]["changes"] if c["node"] == "b"][0]
    assert b_change["upstream"] == "near"
    assert b_change["serial"] == 105
    assert b_change["forward_distance"] == 5


def test_transfer_simulation_distance_tie_by_upstream_name(client):
    # 上游 a 与 z 持相同 serial（距离相同）→ 按上游名取 a
    payload = _simulation_payload(
        nodes=[
            {"name": "s", "serial": 100},
            {"name": "a", "serial": 100},
            {"name": "z", "serial": 100},
            {"name": "b", "serial": 90},
        ],
        edges=[
            {"from": "s", "to": "a"},
            {"from": "s", "to": "z"},
            {"from": "a", "to": "b"},
            {"from": "z", "to": "b"},
        ],
        publish_serial=105,
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    b_change = [c for c in body["rounds"][1]["changes"] if c["node"] == "b"][0]
    assert b_change["upstream"] == "a"
    assert b_change["serial"] == 100
    assert b_change["forward_distance"] == 10


def test_transfer_simulation_non_advance_upstreams_skipped(client):
    # 上游持有的值对 a 是倒退（距离 > 2^31）时不可选取，a 保持旧值且永不到达发布值
    payload = _simulation_payload(
        nodes=[
            {"name": "s", "serial": MAX_SERIAL - 1},
            {"name": "a", "serial": 10},
        ],
        edges=[{"from": "s", "to": "a"}],
        publish_serial=MAX_SERIAL,
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    assert body["converged"] is True
    assert body["sync_rounds_run"] == 0
    assert body["snapshot"] == [
        {"name": "s", "serial": MAX_SERIAL},
        {"name": "a", "serial": 10},
    ]
    # 已稳定（无节点可更新），但 a 的 serial 不等于发布值 → 未收敛节点
    assert body["unconverged"] == [{"name": "a", "serial": 10}]


def test_transfer_simulation_simultaneous_update_uses_previous_round(client):
    # a、b 都只以源 s 为上游，二者在同一轮依据第 0 轮快照同时更新
    payload = _simulation_payload(
        nodes=[
            {"name": "s", "serial": 100},
            {"name": "a", "serial": 90},
            {"name": "b", "serial": 80},
        ],
        edges=[{"from": "s", "to": "a"}, {"from": "s", "to": "b"}],
        publish_serial=110,
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    assert body["converged"] is True
    assert body["sync_rounds_run"] == 1
    assert [(c["node"], c["serial"]) for c in body["rounds"][1]["changes"]] == [
        ("a", 110), ("b", 110)
    ]


def test_transfer_simulation_cycle_converges_same_rule(client):
    # 环路 s->a->b->a：沿同一规则传播后稳定，不会无限循环
    payload = _simulation_payload(
        nodes=[
            {"name": "s", "serial": 100},
            {"name": "a", "serial": 100},
            {"name": "b", "serial": 100},
        ],
        edges=[
            {"from": "s", "to": "a"},
            {"from": "a", "to": "b"},
            {"from": "b", "to": "a"},
        ],
        publish_serial=110,
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    assert body["converged"] is True
    assert [(n["name"], n["serial"]) for n in body["snapshot"]] == [
        ("s", 110), ("a", 110), ("b", 110)
    ]
    assert body["unconverged"] == []


def test_transfer_simulation_self_loop_is_ignored_for_stable_node(client):
    # 自环不能让节点前进：s 发布后 a 无有效上游，保持不变
    payload = _simulation_payload(
        nodes=[{"name": "s", "serial": 100}, {"name": "a", "serial": 90}],
        edges=[{"from": "a", "to": "a"}],
        publish_serial=110,
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    assert body["converged"] is True
    assert body["sync_rounds_run"] == 0
    assert [n["serial"] for n in body["snapshot"]] == [110, 90]
    assert [n["name"] for n in body["unconverged"]] == ["a"]


def test_transfer_simulation_wraparound_advance(client):
    # 源从 2^32-1 发布到 0：相对源为回绕前进（距离 1）；
    # 下游 a 持 2^32-2，0 相对它也是回绕前进（距离 2），可直接同步
    payload = _simulation_payload(
        nodes=[
            {"name": "s", "serial": MAX_SERIAL},
            {"name": "a", "serial": MAX_SERIAL - 1},
        ],
        edges=[{"from": "s", "to": "a"}],
        publish_serial=0,
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    assert body["converged"] is True
    assert body["rounds"][0]["changes"][0]["forward_distance"] == 1
    round1 = body["rounds"][1]["changes"][0]
    assert round1["node"] == "a"
    assert round1["serial"] == 0
    assert round1["forward_distance"] == 2
    assert [(n["name"], n["serial"]) for n in body["snapshot"]] == [
        ("s", 0), ("a", 0)
    ]
    assert body["unconverged"] == []


def test_transfer_simulation_409_when_updatable_after_limit(client):
    # 链长 2、max_rounds=1：第 1 轮后 b 仍可更新 → 409 携带当前快照
    resp = client.post(
        "/api/zones/example.com/transfer-simulations",
        json=_simulation_payload(max_rounds=1),
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "max_rounds_exceeded"
    assert detail["converged"] is False
    assert detail["sync_rounds_run"] == 1
    assert detail["updatable_nodes"] == ["b"]
    assert [(n["name"], n["serial"]) for n in detail["snapshot"]] == [
        ("s", 110), ("a", 110), ("b", 90)
    ]
    assert [n["name"] for n in detail["unconverged"]] == ["b"]
    # 409 同样携带逐轮记录（第 0 轮 + 1 个同步轮）
    assert [r["round"] for r in detail["rounds"]] == [0, 1]


def test_transfer_simulation_converges_exactly_at_limit_is_200(client):
    # 最后一个被允许的同步轮恰好完成收敛（下一轮探测无更新）→ 200
    resp = client.post(
        "/api/zones/example.com/transfer-simulations",
        json=_simulation_payload(max_rounds=2),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["converged"] is True
    assert body["sync_rounds_run"] == 2


def test_transfer_simulation_default_max_rounds(client):
    body = client.post(
        "/api/zones/example.com/transfer-simulations",
        json={
            "nodes": [{"name": "s", "serial": 100}],
            "edges": [],
            "source": "s",
            "publish_serial": 101,
        },
    ).json()
    assert body["max_rounds"] == 64


def test_transfer_simulation_not_persisted(client):
    resp = client.post(
        "/api/zones/example.com/transfer-simulations", json=_simulation_payload()
    )
    assert resp.status_code == 200
    assert client.get("/api/audits").json() == []
    assert client.get("/api/zones/example.com/serial-reservations").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        _simulation_payload(
            edges=[{"from": "ghost", "to": "s"}]),                      # 边引用未声明节点
        _simulation_payload(
            edges=[{"from": "s", "to": "ghost"}]),                      # 边引用未声明节点
        _simulation_payload(source="ghost"),                            # 源节点未声明
        _simulation_payload(
            nodes=[{"name": "s", "serial": 100},
                   {"name": "s", "serial": 90}]),                       # 节点名重复
        _simulation_payload(
            nodes=[{"name": "  ", "serial": 100}]),                     # 空白节点名
        _simulation_payload(
            nodes=[{"name": "s", "serial": -1}]),                       # serial 下越界
        _simulation_payload(
            nodes=[{"name": "s", "serial": MAX_SERIAL + 1}]),           # serial 上越界
        _simulation_payload(source="  "),                               # 空白源节点名
        _simulation_payload(publish_serial=-1),                         # 发布值越界
        _simulation_payload(max_rounds=0),                              # max_rounds 下界
        _simulation_payload(max_rounds=1001),                           # max_rounds 上界
        {"nodes": [], "edges": [], "source": "s",
         "publish_serial": 1},                                          # 空节点列表
        {"edges": [], "source": "s", "publish_serial": 1},              # 缺 nodes
        "not-an-object",                                                # 请求体非对象
    ],
)
def test_transfer_simulation_invalid_body_422(client, payload):
    resp = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    )
    assert resp.status_code == 422


def test_transfer_simulation_invalid_zone_400(client):
    resp = client.post(
        "/api/zones/bad_zone!/transfer-simulations", json=_simulation_payload()
    )
    assert resp.status_code == 400


def test_transfer_simulation_duplicate_edges_treated_as_one(client):
    # 重复传送边不影响结果
    payload = _simulation_payload(
        edges=[{"from": "s", "to": "a"}, {"from": "s", "to": "a"}],
        nodes=[{"name": "s", "serial": 100}, {"name": "a", "serial": 90}],
    )
    body = client.post(
        "/api/zones/example.com/transfer-simulations", json=payload
    ).json()
    assert body["converged"] is True
    assert [(n["name"], n["serial"]) for n in body["snapshot"]] == [
        ("s", 110), ("a", 110)
    ]


# ---------- 多源观测 serial 时间线重建 ----------


def _obs(collector: str, at: str, serial: int) -> dict:
    return {"collector": collector, "observed_at": at, "serial": serial}


def test_serial_timeline_basic_buckets(client):
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("probe-hkg", "2026-09-19T08:00:10Z", 2026091900),
                _obs("probe-fra", "2026-09-19T08:01:00Z", 2026091900),
                _obs("probe-sin", "2026-09-19T08:02:00Z", 2026091890),
                _obs("probe-hkg", "2026-09-19T08:06:00Z", 2026091901),
                _obs("probe-fra", "2026-09-19T08:07:00Z", 2026091901),
                _obs("probe-sin", "2026-09-19T08:08:00Z", 2026091900),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["zone"] == "example.com"
    assert body["bucket_seconds"] == 300
    assert body["timeline"] == [2026091900, 2026091901]
    assert body["total_support"] == 4  # 每桶多数候选各 2 个支持者
    assert body["changes"] == 1
    assert len(body["buckets"]) == 2

    b0, b1 = body["buckets"]
    assert b0["index"] == 0
    assert b0["bucket_start"] == "2026-09-19T08:00:00Z"
    assert b0["bucket_end"] == "2026-09-19T08:05:00Z"
    assert b0["serial"] == 2026091900
    assert b0["supporters"] == ["probe-fra", "probe-hkg"]
    assert b0["dissenters"] == ["probe-sin"]
    # 候选按 serial 升序返回，含支持/异议采集源
    assert [(c["serial"], c["supporters"], c["dissenters"]) for c in b0["candidates"]] == [
        (2026091890, ["probe-sin"], ["probe-fra", "probe-hkg"]),
        (2026091900, ["probe-fra", "probe-hkg"], ["probe-sin"]),
    ]
    assert b1["bucket_start"] == "2026-09-19T08:05:00Z"
    assert b1["serial"] == 2026091901
    assert b1["supporters"] == ["probe-fra", "probe-hkg"]
    assert b1["dissenters"] == ["probe-sin"]
    # 选值序列相邻项仅允许 equal/advance
    assert all(
        classify(a, b) in ("advance", "equal") for a, b in zip(body["timeline"], body["timeline"][1:])
    )


def test_serial_timeline_within_bucket_keeps_latest_per_collector(client):
    # 同一采集源在桶内的更早观测被最新观测覆盖，不产生候选也不成为异议
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("a", "2026-09-19T08:04:59Z", 110),
                _obs("b", "2026-09-19T08:03:00Z", 110),
            ],
        },
    )
    assert resp.status_code == 200
    bucket = resp.json()["buckets"][0]
    assert bucket["candidates"] == [
        {"serial": 110, "supporters": ["a", "b"], "dissenters": []}
    ]
    assert bucket["serial"] == 110
    assert bucket["supporters"] == ["a", "b"]
    assert bucket["dissenters"] == []


def test_serial_timeline_conflicting_latest_same_timestamp_422(client):
    # 同一采集源在同一最新时刻出现不同 serial => 422
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("a", "2026-09-19T08:00:00Z", 101),
                _obs("b", "2026-09-19T08:00:00Z", 100),
            ],
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason"] == "conflicting_latest_observation"
    assert detail["conflicts"] == [
        {
            "bucket_index": 0,
            "bucket_start": "2026-09-19T08:00:00Z",
            "bucket_end": "2026-09-19T08:05:00Z",
            "collector": "a",
            "serials": [100, 101],
        }
    ]


def test_serial_timeline_same_latest_timestamp_same_serial_is_fine(client):
    # 完全重复的观测（同源同时刻同 serial）视为一条，不冲突
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("a", "2026-09-19T08:00:00Z", 100),
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["timeline"] == [100]


def test_serial_timeline_earlier_conflict_superseded_is_fine(client):
    # 早期时刻的冲突值被同源自更新观测整体覆盖 => 不构成 422
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("a", "2026-09-19T08:00:00Z", 101),
                _obs("a", "2026-09-19T08:01:00Z", 102),
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["timeline"] == [102]


def test_serial_timeline_409_regression(client):
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("a", "2026-09-19T08:06:00Z", 50),
            ],
        },
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "unextendable_timeline"
    assert detail["zone"] == "example.com"
    assert detail["bucket_seconds"] == 300
    assert detail["conflict_bucket"] == {
        "index": 1,
        "bucket_start": "2026-09-19T08:05:00Z",
        "bucket_end": "2026-09-19T08:10:00Z",
    }
    blocked = detail["candidates"]
    assert len(blocked) == 1
    assert blocked[0]["serial"] == 50
    assert blocked[0]["supporters"] == ["a"]
    assert blocked[0]["dissenters"] == []
    assert blocked[0]["predecessor_relations"] == [
        {"serial": 100, "relation": "regression"}
    ]
    # 冲突前最优前缀
    assert detail["prefix_timeline"] == [100]
    assert detail["prefix_total_support"] == 1
    assert detail["prefix_changes"] == 0
    assert [b["serial"] for b in detail["prefix_buckets"]] == [100]


def test_serial_timeline_409_incomparable(client):
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 60,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 0),
                _obs("a", "2026-09-19T08:01:00Z", SERIAL_HALF),
            ],
        },
    )
    assert resp.status_code == 409
    relation = resp.json()["detail"]["candidates"][0]["predecessor_relations"][0]
    assert relation == {"serial": 0, "relation": "incomparable"}


def test_serial_timeline_dp_maximizes_total_support(client):
    # 桶0：A=100(支持者 a)、B=101(支持者 b,c)；桶1：仅 B 之后可延续且由 a,b,c 全支持
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("b", "2026-09-19T08:00:00Z", 101),
                _obs("c", "2026-09-19T08:00:00Z", 101),
                _obs("a", "2026-09-19T08:06:00Z", 102),
                _obs("b", "2026-09-19T08:06:00Z", 102),
                _obs("c", "2026-09-19T08:06:00Z", 102),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # 虽然 100 字典序更小，但从 101 起总支持数 2+3=5 > 1+3=4
    assert body["timeline"] == [101, 102]
    assert body["total_support"] == 5


def test_serial_timeline_dp_fewest_changes_tiebreak(client):
    # 两条路径总支持数相同；相等（0 跳变）路径应胜出
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("b", "2026-09-19T08:00:00Z", 100),
                _obs("a", "2026-09-19T08:06:00Z", 100),  # 相等，1 支持
                _obs("b", "2026-09-19T08:06:00Z", 101),  # 前进，1 支持
                _obs("a", "2026-09-19T08:12:00Z", 200),
                _obs("b", "2026-09-19T08:12:00Z", 200),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeline"] == [100, 100, 200]
    assert body["total_support"] == 5  # 2 + 1 + 2
    assert body["changes"] == 1        # 仅 100 -> 200 一跳


def test_serial_timeline_dp_lexicographic_tiebreak(client):
    # 各候选支持数与跳变数完全相同：按整条 serial 序列字典序升序
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 100),
                _obs("b", "2026-09-19T08:00:00Z", 101),
                _obs("a", "2026-09-19T08:06:00Z", 103),
                _obs("b", "2026-09-19T08:06:00Z", 102),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # 可行序列 (100,102)/(100,103)/(101,102)/(101,103) 支持均为 2、跳变均为 1
    assert body["timeline"] == [100, 102]
    assert body["total_support"] == 2
    assert body["changes"] == 1


def test_serial_timeline_wraparound_advance(client):
    # 4294967295 -> 0 为回绕有效前进，允许跨桶
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 60,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", MAX_SERIAL),
                _obs("a", "2026-09-19T08:01:00Z", 0),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeline"] == [MAX_SERIAL, 0]
    assert body["changes"] == 1


def test_serial_timeline_empty_buckets_skipped_and_indexed(client):
    # 观测之间跨过空桶；序号只对非空桶计数
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 1),
                _obs("a", "2026-09-19T08:20:00Z", 2),
            ],
        },
    )
    assert resp.status_code == 200
    buckets = resp.json()["buckets"]
    assert [b["index"] for b in buckets] == [0, 1]
    assert [b["bucket_start"] for b in buckets] == [
        "2026-09-19T08:00:00Z",
        "2026-09-19T08:20:00Z",
    ]


def test_serial_timeline_bucket_boundary_split(client):
    # 08:04:59 与 08:05:00 落入两个 300 秒桶
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 300,
            "observations": [
                _obs("a", "2026-09-19T08:04:59Z", 1),
                _obs("a", "2026-09-19T08:05:00Z", 2),
            ],
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["buckets"]) == 2


def test_serial_timeline_naive_datetime_treated_as_utc(client):
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 60,
            "observations": [_obs("a", "2026-09-19T08:00:00", 7)],
        },
    )
    assert resp.status_code == 200
    bucket = resp.json()["buckets"][0]
    assert bucket["bucket_start"] == "2026-09-19T08:00:00Z"


def test_serial_timeline_single_bucket(client):
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 60,
            "observations": [
                _obs("a", "2026-09-19T08:00:00Z", 1),
                _obs("b", "2026-09-19T08:00:00Z", 1),
                _obs("c", "2026-09-19T08:00:00Z", 2),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeline"] == [1]
    assert body["changes"] == 0
    assert body["total_support"] == 2
    assert body["buckets"][0]["dissenters"] == ["c"]


def test_serial_timeline_not_persisted(client):
    resp = client.post(
        "/api/zones/example.com/serial-timelines",
        json={
            "bucket_seconds": 60,
            "observations": [_obs("a", "2026-09-19T08:00:00Z", 1)],
        },
    )
    assert resp.status_code == 200
    assert client.get("/api/audits").json() == []
    assert client.get("/api/zones/example.com/serial-reservations").json() == []


def test_serial_timeline_invalid_zone_400(client):
    resp = client.post(
        "/api/zones/bad_zone!/serial-timelines",
        json={
            "bucket_seconds": 60,
            "observations": [_obs("a", "2026-09-19T08:00:00Z", 1)],
        },
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"bucket_seconds": 60, "observations": []},                  # 空观测列表
        {"bucket_seconds": 0, "observations": [                      # bucket_seconds 下界
            _obs("a", "2026-09-19T08:00:00Z", 1)]},
        {"bucket_seconds": -5, "observations": [                     # bucket_seconds 负数
            _obs("a", "2026-09-19T08:00:00Z", 1)]},
        {"observations": [                                           # 缺 bucket_seconds
            _obs("a", "2026-09-19T08:00:00Z", 1)]},
        {"bucket_seconds": 60, "observations": [                    # 缺 collector
            {"observed_at": "2026-09-19T08:00:00Z", "serial": 1}]},
        {"bucket_seconds": 60, "observations": [                    # 空白 collector
            _obs("   ", "2026-09-19T08:00:00Z", 1)]},
        {"bucket_seconds": 60, "observations": [                    # 缺 observed_at
            {"collector": "a", "serial": 1}]},
        {"bucket_seconds": 60, "observations": [                    # 时间格式非法
            _obs("a", "not-a-time", 1)]},
        {"bucket_seconds": 60, "observations": [                    # serial 下越界
            _obs("a", "2026-09-19T08:00:00Z", -1)]},
        {"bucket_seconds": 60, "observations": [                    # serial 上越界
            _obs("a", "2026-09-19T08:00:00Z", MAX_SERIAL + 1)]},
        {"bucket_seconds": 60},                                     # 缺 observations
        {"bucket_seconds": 60, "observations": [                    # 观测非对象
            "not-an-object"]},
        "not-an-object",                                            # 请求体非对象
    ],
)
def test_serial_timeline_invalid_body_422(client, payload):
    resp = client.post(
        "/api/zones/example.com/serial-timelines", json=payload
    )
    assert resp.status_code == 422


# ---------- 多权威节点 RRset 快照一致性检查 ----------


def _rr(owner, ttl, rdata, rrtype="A", rrclass="IN") -> dict:
    return {"owner": owner, "type": rrtype, "class": rrclass, "ttl": ttl, "rdata": rdata}


def _snapshot_payload(nodes, min_agreement=2) -> dict:
    return {"min_agreement": min_agreement, "nodes": nodes}


def _node(name, *rrsets) -> dict:
    return {"name": name, "rrsets": list(rrsets)}


def test_snapshot_consistency_full_consensus_200(client):
    payload = _snapshot_payload(
        [
            _node("ns1", _rr("www.example.com", 300, ["192.0.2.1", "192.0.2.2"])),
            _node("ns2", _rr("www.example.com", 300, ["192.0.2.1", "192.0.2.2"])),
        ]
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["zone"] == "example.com"
    assert body["node_count"] == 2
    assert body["min_agreement"] == 2
    assert body["conflicts"] == []
    snapshot = body["consensus_snapshot"]
    assert len(snapshot) == 1
    assert snapshot[0] == {
        "owner": "www.example.com",
        "type": "A",
        "class": "IN",
        "ttl": 300,
        "rdata": ["192.0.2.1", "192.0.2.2"],
        "agreed_nodes": ["ns1", "ns2"],
        "dissenting_nodes": [],
    }
    # 全部一致时逐节点差异为空
    assert body["node_diffs"] == [
        {"node": "ns1", "missing": [], "extra": [], "mismatched": []},
        {"node": "ns2", "missing": [], "extra": [], "mismatched": []},
    ]
    # 结果不落库
    assert client.get("/api/audits").json() == []
    assert client.get("/api/zones/example.com/serial-reservations").json() == []


def test_snapshot_consistency_normalizes_owner_type_and_rdata_order(client):
    # owner 大小写/末尾根点/@、type/class 大小写、rdata 顺序与重复项全部规范化后一致
    payload = _snapshot_payload(
        [
            _node(
                "ns1",
                _rr("WWW.Example.COM.", 300, ["192.0.2.2", "192.0.2.1"], rrtype="a", rrclass="in"),
                _rr("@", 3600, ["ns2.example.com."], rrtype="NS"),
            ),
            _node(
                "ns2",
                _rr("www.example.com", 300, ["192.0.2.1", "192.0.2.1", "192.0.2.2"]),
                _rr("example.com.", 3600, ["ns2.example.com."], rrtype="ns"),
            ),
        ]
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    snapshot = resp.json()["consensus_snapshot"]
    assert [(rr["owner"], rr["type"], rr["class"], rr["rdata"]) for rr in snapshot] == [
        ("example.com", "NS", "IN", ["ns2.example.com."]),
        ("www.example.com", "A", "IN", ["192.0.2.1", "192.0.2.2"]),
    ]


def test_snapshot_consistency_majority_wins_with_dissenters(client):
    # 3 节点中 2 个持有同一变体，第 3 个 TTL 不同 => 多数变体进入共识并给出 mismatched
    payload = _snapshot_payload(
        [
            _node("ns1", _rr("www", 300, ["192.0.2.1"])),
            _node("ns2", _rr("www", 300, ["192.0.2.1"])),
            _node("ns3", _rr("www", 60, ["192.0.2.1"])),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    body = resp.json()
    rrset = body["consensus_snapshot"][0]
    assert rrset["ttl"] == 300
    assert rrset["agreed_nodes"] == ["ns1", "ns2"]
    assert rrset["dissenting_nodes"] == ["ns3"]
    diff_ns3 = next(d for d in body["node_diffs"] if d["node"] == "ns3")
    assert diff_ns3["missing"] == []
    assert diff_ns3["extra"] == []
    mismatch = diff_ns3["mismatched"]
    assert len(mismatch) == 1
    assert mismatch[0]["owner"] == "www"
    assert mismatch[0]["consensus"]["ttl"] == 300
    assert mismatch[0]["node_rrset"]["ttl"] == 60
    assert mismatch[0]["node_rrset"]["rdata"] == ["192.0.2.1"]
    # 其余节点无差异
    for name in ("ns1", "ns2"):
        diff = next(d for d in body["node_diffs"] if d["node"] == name)
        assert diff == {"node": name, "missing": [], "extra": [], "mismatched": []}


def test_snapshot_consistency_rdata_difference_is_mismatch(client):
    payload = _snapshot_payload(
        [
            _node("ns1", _rr("www", 300, ["192.0.2.1"])),
            _node("ns2", _rr("www", 300, ["192.0.2.2"])),
            _node("ns3", _rr("www", 300, ["192.0.2.2"])),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["consensus_snapshot"][0]["rdata"] == ["192.0.2.2"]
    diff_ns1 = next(d for d in body["node_diffs"] if d["node"] == "ns1")
    mismatch = diff_ns1["mismatched"][0]
    assert mismatch["consensus"]["rdata"] == ["192.0.2.2"]
    assert mismatch["node_rrset"]["rdata"] == ["192.0.2.1"]


def test_snapshot_consensus_tiebreak_rdata_then_ttl(client):
    # 2 节点各持一个变体且同意门槛为 1：按 rdata 序列字典序升序决出确定结果
    payload = _snapshot_payload(
        [
            _node("a", _rr("h", 300, ["192.0.2.9"])),
            _node("b", _rr("h", 300, ["192.0.2.1"])),
        ],
        min_agreement=1,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    assert resp.json()["consensus_snapshot"][0]["rdata"] == ["192.0.2.1"]


def test_snapshot_consensus_tiebreak_lower_ttl_when_rdata_equal(client):
    # rdata 相同、TTL 不同且支持数并列时取较小 TTL
    payload = _snapshot_payload(
        [
            _node("a", _rr("h", 600, ["1.1.1.1"])),
            _node("b", _rr("h", 300, ["1.1.1.1"])),
        ],
        min_agreement=1,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    assert resp.json()["consensus_snapshot"][0]["ttl"] == 300


def test_snapshot_consistency_min_agreement_one_with_disjoint_keys(client):
    # 门槛为 1 时各节点独有键都进共识，并互相记 missing
    payload = _snapshot_payload(
        [
            _node("ns1", _rr("a", 300, ["1.1.1.1"])),
            _node("ns2", _rr("b", 60, ["2.2.2.2"])),
        ],
        min_agreement=1,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [rr["owner"] for rr in body["consensus_snapshot"]] == ["a", "b"]
    assert body["consensus_snapshot"][0]["dissenting_nodes"] == ["ns2"]
    diff_ns1, diff_ns2 = body["node_diffs"]
    assert diff_ns1["missing"] == [
        {"owner": "b", "type": "A", "class": "IN", "ttl": 60, "rdata": ["2.2.2.2"]}
    ]
    assert diff_ns2["missing"] == [
        {"owner": "a", "type": "A", "class": "IN", "ttl": 300, "rdata": ["1.1.1.1"]}
    ]


def test_snapshot_consistency_empty_snapshots(client):
    payload = _snapshot_payload(
        [_node("ns1"), _node("ns2", )], min_agreement=2
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["consensus_snapshot"] == []
    assert all(d["missing"] == [] and d["extra"] == [] and d["mismatched"] == []
               for d in body["node_diffs"])


def test_snapshot_consistency_409_lists_conflicts_and_partial_consensus(client):
    # NS 三节点一致进共识；MX 仅 1 节点持有 => 409
    payload = _snapshot_payload(
        [
            _node(
                "ns1",
                _rr("@", 3600, ["ns1.example.com.", "ns2.example.com."], rrtype="NS"),
            ),
            _node(
                "ns2",
                _rr("@", 3600, ["ns1.example.com.", "ns2.example.com."], rrtype="NS"),
                _rr("@", 600, ["10 mx1.example.com."], rrtype="MX"),
            ),
            _node(
                "ns3",
                _rr("@", 3600, ["ns1.example.com.", "ns2.example.com."], rrtype="NS"),
            ),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["reason"] == "insufficient_consensus"
    assert detail["zone"] == "example.com"
    assert detail["node_count"] == 3
    assert detail["min_agreement"] == 2
    assert len(detail["conflicts"]) == 1
    conflict = detail["conflicts"][0]
    assert conflict == {
        "owner": "example.com",
        "type": "MX",
        "class": "IN",
        "best_agreement": 1,
        "variants": [
            {"ttl": 600, "rdata": ["10 mx1.example.com."], "nodes": ["ns2"]}
        ],
        "missing_nodes": ["ns1", "ns3"],
    }
    # 部分共识快照与逐节点差异仍然返回；未达门槛的键计入持有节点 extra
    partial = detail["consensus_snapshot"]
    assert len(partial) == 1
    assert partial[0]["owner"] == "example.com"
    assert partial[0]["type"] == "NS"
    assert partial[0]["agreed_nodes"] == ["ns1", "ns2", "ns3"]
    diffs = {d["node"]: d for d in detail["node_diffs"]}
    assert diffs["ns2"]["extra"] == [
        {"owner": "example.com", "type": "MX", "class": "IN",
         "ttl": 600, "rdata": ["10 mx1.example.com."]}
    ]
    assert diffs["ns1"]["extra"] == [] and diffs["ns1"]["missing"] == []
    assert diffs["ns3"]["extra"] == [] and diffs["ns3"]["missing"] == []


def test_snapshot_consistency_409_multiple_conflicts_sorted(client):
    # 3 个冲突键（含不同 type/class），按 owner/type/class 排序返回
    payload = _snapshot_payload(
        [
            _node(
                "ns1",
                _rr("z", 300, ["1.1.1.1"]),
                _rr("a", 300, ["10 mail"], rrtype="MX"),
                _rr("a", 300, ["zone."], rrtype="SOA"),
            ),
            _node("ns2"),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 409
    conflicts = resp.json()["detail"]["conflicts"]
    assert [(c["owner"], c["type"], c["class"]) for c in conflicts] == [
        ("a", "MX", "IN"),
        ("a", "SOA", "IN"),
        ("z", "A", "IN"),
    ]
    for conflict in conflicts:
        assert conflict["missing_nodes"] == ["ns2"]


def test_snapshot_consistency_409_variants_descending_by_support(client):
    # 同一 RRset 出现三个不同变体（不同 rdata），按支持数降序
    payload = _snapshot_payload(
        [
            _node("ns1", _rr("www", 300, ["192.0.2.1"])),
            _node("ns2", _rr("www", 300, ["192.0.2.2"])),
            _node("ns3", _rr("www", 300, ["192.0.2.3"])),
            _node("ns4", _rr("www", 300, ["192.0.2.1"])),
        ],
        min_agreement=3,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 409
    conflict = resp.json()["detail"]["conflicts"][0]
    assert conflict["best_agreement"] == 2
    variants = conflict["variants"]
    assert [v["rdata"] for v in variants] == [
        ["192.0.2.1"],
        ["192.0.2.2"],
        ["192.0.2.3"],
    ]
    assert variants[0]["nodes"] == ["ns1", "ns4"]
    assert variants[1]["nodes"] == ["ns2"]
    assert variants[2]["nodes"] == ["ns3"]
    assert conflict["missing_nodes"] == []


def test_snapshot_consistency_threshold_boundary(client):
    # 恰好达到门槛 => 200；门槛 +1 => 409
    base_nodes = [
        _node("ns1", _rr("www", 300, ["1.1.1.1"])),
        _node("ns2", _rr("www", 300, ["1.1.1.1"])),
        _node("ns3", _rr("www", 300, ["2.2.2.2"])),
    ]
    ok = client.post(
        "/api/zones/example.com/snapshot-consistency",
        json=_snapshot_payload(base_nodes, min_agreement=2),
    )
    assert ok.status_code == 200
    bad = client.post(
        "/api/zones/example.com/snapshot-consistency",
        json=_snapshot_payload(base_nodes, min_agreement=3),
    )
    assert bad.status_code == 409
    assert bad.json()["detail"]["conflicts"][0]["best_agreement"] == 2


def test_snapshot_consistency_duplicate_rrset_identical_is_idempotent(client):
    # 同一节点内完全相同（规范化后）的重复 RRset 按一条处理
    payload = _snapshot_payload(
        [
            _node(
                "ns1",
                _rr("WWW.", 300, ["1.1.1.1"]),
                _rr("www", 300, ["1.1.1.1", "1.1.1.1"]),
            ),
            _node("ns2", _rr("www", 300, ["1.1.1.1"])),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 200
    assert len(resp.json()["consensus_snapshot"]) == 1


def test_snapshot_consistency_contradictory_rrset_same_node_422(client):
    # 同一节点对同一键给出 TTL 不同的两个变体 => 422
    payload = _snapshot_payload(
        [
            _node(
                "ns1",
                _rr("www.", 300, ["1.1.1.1"]),
                _rr("www", 60, ["1.1.1.1"]),
            ),
            _node("ns2", _rr("www", 300, ["1.1.1.1"])),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason"] == "duplicate_rrset_in_snapshot"
    assert detail["conflicts"] == [
        {"node": "ns1", "owner": "www", "type": "A", "class": "IN"}
    ]


def test_snapshot_consistency_rdata_is_case_sensitive(client):
    # RDATA 文本大小写敏感：不同大小写视为不同变体
    payload = _snapshot_payload(
        [
            _node("ns1", _rr("txt", 300, ["Hello"], rrtype="TXT")),
            _node("ns2", _rr("txt", 300, ["hello"], rrtype="TXT")),
        ],
        min_agreement=2,
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 409
    variants = resp.json()["detail"]["conflicts"][0]["variants"]
    assert [v["rdata"] for v in variants] == [["Hello"], ["hello"]]


def test_snapshot_consistency_not_persisted_on_409(client):
    payload = _snapshot_payload(
        [_node("ns1", _rr("a", 1, ["1"])), _node("ns2")], min_agreement=2
    )
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 409
    assert client.get("/api/audits").json() == []
    assert client.get("/api/zones/example.com/serial-reservations").json() == []


def test_snapshot_consistency_invalid_zone_400(client):
    resp = client.post(
        "/api/zones/bad_zone!/snapshot-consistency",
        json=_snapshot_payload([_node("ns1")], min_agreement=1),
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        _snapshot_payload([], min_agreement=1),                         # 空节点列表
        _snapshot_payload([_node("ns1")], min_agreement=2),            # 门槛超过节点数
        _snapshot_payload([_node("ns1")], min_agreement=0),            # 门槛下界
        {"nodes": [_node("ns1")]},                                     # 缺 min_agreement
        {"min_agreement": 1},                                         # 缺 nodes
        _snapshot_payload(                                             # 节点名重复（规范化空白后）
            [_node(" ns1 "), _node("ns1")], min_agreement=1),
        _snapshot_payload([_node("   ")], min_agreement=1),            # 空白节点名
        _snapshot_payload(                                             # 非法 owner
            [_node("ns1", _rr("bad owner", 1, ["x"]))], min_agreement=1),
        _snapshot_payload(                                             # TTL 越界
            [_node("ns1", _rr("a", -1, ["x"]))], min_agreement=1),
        _snapshot_payload(                                             # TTL 上越界
            [_node("ns1", _rr("a", 2**31, ["x"]))], min_agreement=1),
        _snapshot_payload(                                             # rdata 为空
            [_node("ns1", _rr("a", 1, []))], min_agreement=1),
        _snapshot_payload(                                             # rdata 空白条目
            [_node("ns1", _rr("a", 1, ["   "]))], min_agreement=1),
        _snapshot_payload(                                             # 非法 type
            [_node("ns1", {"owner": "a", "type": "1A", "class": "IN",
                           "ttl": 1, "rdata": ["x"]})], min_agreement=1),
        _snapshot_payload(                                             # 非法 class
            [_node("ns1", _rr("a", 1, ["x"], rrclass="1N"))], min_agreement=1),
        "not-an-object",                                              # 请求体非对象
    ],
)
def test_snapshot_consistency_invalid_body_422(client, payload):
    resp = client.post(
        "/api/zones/example.com/snapshot-consistency", json=payload
    )
    assert resp.status_code == 422


def test_snapshot_normalize_owner_unit():
    from app.snapshot import normalize_owner

    assert normalize_owner("@", "Example.COM") == "example.com"
    assert normalize_owner("WWW.Example.COM.", "example.com") == "www.example.com"
    assert normalize_owner("  root.  ", "example.com") == "root"
    assert normalize_owner(".", "example.com") == "."


# ---------- 并发：串行化分配，不重复 serial ----------

def test_concurrent_reservations_no_duplicate_serials(tmp_path):
    db_path = str(tmp_path / "concurrent.db")
    db.init_db(db_path)
    base = _utc_today_base()
    workers = 8
    attempts = 5  # 8*5 = 40，不超过当日 99 个可用编号

    allocated: list[int] = []
    lock = threading.Lock()

    def worker(worker_id: int) -> None:
        current = base
        for attempt in range(attempts):
            request_id = f"w{worker_id}-a{attempt}"
            for _ in range(60):  # 409 时按 latest_serial 追赶重试
                try:
                    record, replayed = db.reserve_serial(
                        db_path, "example.com", request_id, current
                    )
                    break
                except db.ReservationError as exc:
                    if exc.code == db.ERR_STALE_CURRENT and exc.latest_serial is not None:
                        current = exc.latest_serial
                        continue
                    raise
                except sqlite3.OperationalError:
                    continue
            else:
                raise AssertionError("并发预留多次重试后仍失败")
            assert not replayed
            with lock:
                allocated.append(record["serial"])
            current = record["serial"]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(allocated) == workers * attempts
    assert len(set(allocated)) == len(allocated)  # 无重复
    assert set(allocated) == {base + i for i in range(1, workers * attempts + 1)}


def test_concurrent_same_request_id_races_to_one_reservation(tmp_path):
    db_path = str(tmp_path / "concurrent-idem.db")
    with TestClient(create_app(db_path=db_path)) as c:
        base = _utc_today_base()
        results: list[tuple[int, int]] = []
        barrier = threading.Barrier(8)

        def hit() -> None:
            barrier.wait()
            resp = c.post(
                "/api/zones/example.com/serial-reservations",
                json={"current": base, "request_id": "same-req"},
            )
            results.append((resp.status_code, resp.json()["serial"]))

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert {status for status, _ in results} == {200, 201}
        assert sum(1 for status, _ in results if status == 201) == 1
        assert sum(1 for status, _ in results if status == 200) == 7
        assert {serial for _, serial in results} == {base + 1}
        rows = c.get("/api/zones/example.com/serial-reservations").json()
        assert len(rows) == 1
