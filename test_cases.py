"""
test_cases.py — Hardcoded test cases from Sovra LLM Testcase document

Usage:
    python test_cases.py          # run all 10
    python test_cases.py TC-001   # run single case
"""

import sys
import json
import httpx

RAG_URL = "http://localhost:8000/inspect"

TEST_CASES = {
    "TC-001": {
        "description": "Recurring BMFO at AOI-04 — board-level or station contamination?",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "BMFO",
                    "defect_name": "Base Material Foreign Object",
                    "confidence": 0.9981,
                    "severity_hint": "medium",
                    "location_description": "center area of local AOI crop"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "medium",
                "same_defect_count_30d": 9,
                "same_defect_count_90d": 21,
                "trend": "increasing",
                "most_common_shift": "Afternoon",
                "recurring_location": "center_or_large_crop_area",
                "common_repair_action": "cleaned affected board and reinspected",
                "recurrence_after_repair": True,
                "maintenance_note": "minor dust accumulation near board loading area",
                "cleaning_status": "temporary cleaning done — full station cleaning not done",
                "supplier_quality_status": "normal"
            }
        },
        "expected_action": "clean_station"
    },

    "TC-002": {
        "description": "Recurring Open defects — previous rework failed, same functional region",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "OP",
                    "defect_name": "Open",
                    "confidence": 0.9973,
                    "severity_hint": "critical",
                    "location_description": "center and lower center — BCM-01 power distribution path"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "high",
                "same_defect_count_90d": 12,
                "recurring_location": "center_and_lower_center_crop_regions",
                "functional_zone": True,
                "board_zone": "BCM-01 power distribution conductor path",
                "common_repair_action": "manual copper bridge repair",
                "recurrence_after_repair": True,
                "last_repair_result": "passed continuity test but defect recurred after 6 days",
                "previous_dispositions": "passed_after_rework"
            }
        },
        "expected_action": "escalate"
    },

    "TC-003": {
        "description": "Spurious Copper after AOI recalibration — possible false positive",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "SC",
                    "defect_name": "Spurious Copper",
                    "confidence": 0.9951,
                    "severity_hint": "medium_high",
                    "location_description": "reflective copper trace region"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "medium",
                "calibration_event": "AOI camera and lighting recalibration on 2026-05-15",
                "detection_rate_change": "SC rate increased 38% after calibration; manual confirmation rate only 52%",
                "supplier_quality_status": "normal"
            }
        },
        "expected_action": "manual_review"
    },

    "TC-004": {
        "description": "Conductor Scratch recurring after handling robot repair",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "CS",
                    "defect_name": "Conductor Scratch",
                    "confidence": 0.9999,
                    "severity_hint": "high",
                    "location_description": "fine conductor traces near handling path"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "medium_high",
                "same_defect_count_30d": 5,
                "trend": "recurring after repair",
                "common_repair_action": "polished contact surface and replaced handling guide",
                "recurrence_after_repair": True,
                "last_repair_result": "visual check passed but defect recurred",
                "maintenance_note": "no full robot recalibration record found after guide replacement",
                "previous_dispositions": "manual_review_pending"
            }
        },
        "expected_action": "escalate"
    },

    "TC-005": {
        "description": "No defect detected — new Rev-C material qualification batch",
        "payload": {
            "vision_result": {
                "defect_detected": False,
                "primary_detection": None,
                "detections": []
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100 Rev-C",
                "safety_criticality": "medium",
                "same_defect_count_30d": 0,
                "same_defect_count_90d": 0,
                "supplier_id": "SUP-077",
                "supplier_quality_status": "new_qualification_supplier",
                "batch_id": "BATCH-REV-C-QUAL-001"
            }
        },
        "expected_action": "pass"
    },

    "TC-006": {
        "description": "Short defects linked to same supplier lot across multiple lines and factories",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "SH",
                    "defect_name": "Short",
                    "confidence": 0.9726,
                    "severity_hint": "critical",
                    "location_description": "dense conductor routing area"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "high",
                "same_defect_count_30d": 18,
                "trend": "increasing",
                "affected_lines": "LINE-A3, LINE-B1",
                "affected_factories": "FACTORY-BERLIN-01, FACTORY-DRESDEN-02",
                "supplier_id": "SUP-042",
                "supplier_quality_status": "open_quality_case",
                "supplier_lot": "LOT-CU-2026-0518-77",
                "quarantine_status": "pending",
                "previous_dispositions": "7 rejected, 3 rework_failed out of last 10 cases"
            }
        },
        "expected_action": "quarantine_lot"
    },

    "TC-007": {
        "description": "CFO night shift — dust accumulation, temporary cleaning not enough",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "CFO",
                    "defect_name": "Conductor Foreign Object",
                    "confidence": 0.9682,
                    "severity_hint": "high",
                    "location_description": "conductor region, multiple areas"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "medium",
                "shift": "Night",
                "same_defect_count_30d": 6,
                "trend": "increasing during night shift",
                "most_common_shift": "Night",
                "maintenance_note": "dust accumulation near board loading area",
                "cleaning_status": "temporary cleaning done — full cleaning not done",
                "supplier_quality_status": "normal"
            }
        },
        "expected_action": "clean_station"
    },

    "TC-008": {
        "description": "Short + Open on same board — multiple critical defects, previous rework failed",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "SH",
                    "defect_name": "Short",
                    "confidence": 0.6612,
                    "severity_hint": "critical",
                    "location_description": "body control PCB, multiple regions"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "high",
                "same_defect_count_30d": 3,
                "supplier_quality_status": "under_review",
                "common_repair_action": "engineering rework attempted",
                "recurrence_after_repair": True,
                "last_repair_result": "engineering rework failed",
                "previous_dispositions": "2 rejected, 1 engineering_rework_failed out of last 3 cases"
            }
        },
        "expected_action": "reject"
    },

    "TC-009": {
        "description": "Low confidence Mouse Bite — no history, uncertain detection",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "MB",
                    "defect_name": "Mouse Bite",
                    "confidence": 0.3375,
                    "severity_hint": "medium",
                    "location_description": "conductor edge area"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-C2",
                "station_id": "AOI-09",
                "product_model": "PCB-X200",
                "safety_criticality": "medium",
                "same_defect_count_30d": 0,
                "same_defect_count_90d": 0,
                "trend": "no history",
                "supplier_quality_status": "normal"
            }
        },
        "expected_action": "manual_review"
    },

    "TC-010": {
        "description": "High confidence Hole Breakout — but in non-functional test coupon zone",
        "payload": {
            "vision_result": {
                "defect_detected": True,
                "primary_detection": {
                    "defect_code": "HB",
                    "defect_name": "Hole Breakout",
                    "confidence": 0.9765,
                    "severity_hint": "high",
                    "location_description": "test coupon area — non-functional zone"
                }
            },
            "production_context": {
                "factory_name": "Berlin Electronics Plant",
                "production_line_id": "LINE-A3",
                "station_id": "AOI-04",
                "product_model": "PCB-X100",
                "safety_criticality": "medium",
                "functional_zone": False,
                "board_zone": "test_coupon_area — process monitoring only, not functional circuit",
                "same_defect_count_30d": 4,
                "trend": "stable",
                "supplier_quality_status": "normal"
            }
        },
        "expected_action": "pass"
    },
}


def run_case(case_id: str) -> None:
    case = TEST_CASES[case_id]
    print(f"\n{'='*60}")
    print(f"  {case_id}: {case['description']}")
    print(f"  Expected : {case['expected_action']}")
    print(f"{'='*60}")

    try:
        response = httpx.post(RAG_URL, json=case["payload"], timeout=300)
        response.raise_for_status()
        result = response.json()

        action = result.get("recommended_action", "?")
        match = "PASS" if action == case["expected_action"] else "MISMATCH"

        print(f"  Got      : {action}  [{match}]")
        print(f"  Severity : {result.get('severity_assessment', '?')}")
        print(f"  Time     : {result.get('inspection_timestamp', '?')}")
        print(f"  Explain  : {result.get('defect_explanation', '')[:100]}...")
        print(f"  Justify  : {result.get('justification', '')[:100]}...")
        print(f"  Refs     : {result.get('sop_references', [])}")

    except httpx.ConnectError:
        print(f"  ERROR: RAG API not running at {RAG_URL}")
    except httpx.HTTPStatusError as e:
        print(f"  ERROR {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        case_id = sys.argv[1].upper()
        if case_id not in TEST_CASES:
            print(f"Unknown: {case_id}. Available: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)
        run_case(case_id)
    else:
        print(f"Running {len(TEST_CASES)} test cases against {RAG_URL}")
        for case_id in TEST_CASES:
            run_case(case_id)
        print(f"\n{'='*60}\nDone.\n")
