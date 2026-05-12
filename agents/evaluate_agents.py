import time
from statistics import mean
from typing import List, Dict

from app.main import run_uab
from agents.rag_agent import run_rag_agent
from agents.sql_agent import load_excel_to_sqlite, run_sql_agent
from agents.web_agent import run_web_agent


ROUTE_TEST_CASES = [
    {
        "question": "What is the minimum attendance required to appear in exams?",
        "expected_agent": "rag"
    },
    {
        "question": "Who has the highest CGPA in Electronics department?",
        "expected_agent": "sql"
    },
    {
        "question": "Any internships available for 3rd year CSE students?",
        "expected_agent": "web"
    },
]

RAG_TEST_CASES = [
    {
        "question": "What are the prerequisites for the M.Tech program?",
        "expected_keywords": ["prerequisite", "M.Tech", "eligibility"]
    },
    {
        "question": "What is the attendance policy for exams?",
        "expected_keywords": ["attendance", "exam", "minimum"]
    },
]

SQL_TEST_CASES = [
    {
        "question": "Who has the highest CGPA overall?",
        "expected_keywords": ["CGPA", "highest", "top"]
    },
    {
        "question": "Which students have attendance below 75%?",
        "expected_keywords": ["attendance", "below", "75"]
    },
]

WEB_TEST_CASES = [
    {
        "question": "What scholarships are available for engineering students?",
        "expected_keywords": ["scholarship", "engineering", "students"]
    },
    {
        "question": "Internship opportunities for computer science students in India",
        "expected_keywords": ["internship", "computer science", "India"]
    },
]


def normalized(text: str) -> str:
    return text.strip().lower() if text else ""


def contains_keywords(answer: str, keywords: List[str]) -> bool:
    lower = normalized(answer)
    return any(keyword.lower() in lower for keyword in keywords)


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def eval_route() -> Dict[str, object]:
    route_results = []

    for case in ROUTE_TEST_CASES:
        result, latency = time_call(run_uab, case["question"])
        route_results.append({
            "question": case["question"],
            "expected": case["expected_agent"],
            "predicted": result.get("agent", ""),
            "latency": latency,
            "success": result.get("agent", "") == case["expected_agent"],
            "answer": result.get("answer", ""),
        })

    accuracy = mean(1.0 if r["success"] else 0.0 for r in route_results)
    avg_latency = mean(r["latency"] for r in route_results)

    return {
        "stage": "route",
        "num_cases": len(route_results),
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "results": route_results,
    }


def eval_rag() -> Dict[str, object]:
    rag_results = []

    for case in RAG_TEST_CASES:
        result, latency = time_call(run_rag_agent, case["question"])
        success = bool(result.get("sources")) and contains_keywords(result.get("answer", ""), case["expected_keywords"])
        rag_results.append({
            "question": case["question"],
            "latency": latency,
            "answer_length": len(result.get("answer", "")),
            "sources": result.get("sources", []),
            "success": success,
            "answer": result.get("answer", ""),
        })

    avg_latency = mean(r["latency"] for r in rag_results)
    source_rate = mean(1.0 if r["sources"] else 0.0 for r in rag_results)
    success_rate = mean(1.0 if r["success"] else 0.0 for r in rag_results)

    return {
        "stage": "rag",
        "num_cases": len(rag_results),
        "avg_latency": avg_latency,
        "source_rate": source_rate,
        "success_rate": success_rate,
        "results": rag_results,
    }


def eval_sql() -> Dict[str, object]:
    sql_results = []

    for case in SQL_TEST_CASES:
        result, latency = time_call(run_sql_agent, case["question"])
        success = result.get("success", False) and contains_keywords(result.get("answer", ""), case["expected_keywords"])
        sql_results.append({
            "question": case["question"],
            "latency": latency,
            "answer_length": len(result.get("answer", "")),
            "success": success,
            "error": result.get("error"),
            "answer": result.get("answer", ""),
        })

    avg_latency = mean(r["latency"] for r in sql_results)
    success_rate = mean(1.0 if r["success"] else 0.0 for r in sql_results)

    return {
        "stage": "sql",
        "num_cases": len(sql_results),
        "avg_latency": avg_latency,
        "success_rate": success_rate,
        "results": sql_results,
    }


def eval_web() -> Dict[str, object]:
    web_results = []

    for case in WEB_TEST_CASES:
        result, latency = time_call(run_web_agent, case["question"])
        success = bool(result.get("sources")) and contains_keywords(result.get("answer", ""), case["expected_keywords"])
        web_results.append({
            "question": case["question"],
            "latency": latency,
            "answer_length": len(result.get("answer", "")),
            "sources": result.get("sources", []),
            "success": success,
            "error": result.get("error"),
            "answer": result.get("answer", ""),
        })

    avg_latency = mean(r["latency"] for r in web_results)
    source_rate = mean(1.0 if r["sources"] else 0.0 for r in web_results)
    success_rate = mean(1.0 if r["success"] else 0.0 for r in web_results)

    return {
        "stage": "web",
        "num_cases": len(web_results),
        "avg_latency": avg_latency,
        "source_rate": source_rate,
        "success_rate": success_rate,
        "results": web_results,
    }


def print_report(report: Dict[str, object]):
    print(f"\n=== {report['stage'].upper()} REPORT ===")
    print(f"Cases: {report['num_cases']}")
    print(f"Average latency: {report['avg_latency']:.2f}s")
    if report['stage'] == 'route':
        print(f"Route accuracy: {report['accuracy']:.2%}")
    else:
        print(f"Success rate: {report['success_rate']:.2%}")
        if 'source_rate' in report:
            print(f"Source coverage: {report['source_rate']:.2%}")

    print("\nDetails:")
    for result in report['results']:
        print(f"- Q: {result['question']}")
        print(f"  Latency: {result['latency']:.2f}s")
        if report['stage'] == 'route':
            print(f"  Expected: {result['expected']}, Predicted: {result['predicted']}")
            print(f"  Success: {result['success']}")
        else:
            print(f"  Success: {result['success']}")
            if 'sources' in result:
                print(f"  Sources: {result['sources']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
        print(f"  Answer: {result['answer'][:260].replace('\n',' ')}")
        print("")


def evaluate_all():
    print("Starting evaluation of UAB agents. This may take a few minutes.")
    load_excel_to_sqlite()
    route_report = eval_route()
    rag_report = eval_rag()
    sql_report = eval_sql()
    web_report = eval_web()

    print_report(route_report)
    print_report(rag_report)
    print_report(sql_report)
    print_report(web_report)

    return {
        "route": route_report,
        "rag": rag_report,
        "sql": sql_report,
        "web": web_report,
    }


if __name__ == "__main__":
    evaluate_all()
