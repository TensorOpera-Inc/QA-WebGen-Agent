#!/usr/bin/env python3
"""
Evaluate Deployed Web URLs
==========================
This script evaluates a list of deployed web URLs using WebGen-Agent's feedback capabilities.

Input Format (JSON file or inline):
    [
        {"prompt": "Build a todo app...", "url": "https://example1.com"},
        {"prompt": "Create a dashboard...", "url": "https://example2.com"}
    ]

Output Format (JSON):
    {
        "results": [
            {
                "prompt": "...",
                "url": "...",
                "screenshot_path": "...",
                "screenshot_analysis": {...},
                "screenshot_grade": 4,
                "webvoyager_feedback": {...},  // if enabled
                "webvoyager_grade": 5,         // if enabled
                "status": "success" | "error",
                "error_message": null
            }
        ],
        "summary": {
            "total": 10,
            "success": 8,
            "failed": 2,
            "avg_screenshot_grade": 3.5,
            "avg_webvoyager_grade": 4.0  // if enabled
        }
    }

Usage:
    python evaluate_deployed_urls.py --input tasks.json --output results.json --vlm-model gpt-4o
    python evaluate_deployed_urls.py --input tasks.json --output results.json --enable-webvoyager
"""

import os
import sys
import json
import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Setup project path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Try to use webdriver-manager if available, otherwise fall back to system chromedriver
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False
    print("[Warning] webdriver-manager not installed. Using system chromedriver.")
    print("[Warning] Install with: pip install webdriver-manager")

from utils.vlm_generation import vlm_generation
from utils.llm_fb_generation import llm_fb_generation
from utils.get_screenshot_description import (
    get_screenshot_description,
    get_screenshot_grade,
    encode_image
)
from utils.get_webvoyager_feedback import get_webvoyager_feedback
from utils.generate_gui_agent_instruction import generate_gui_agent_instruction


def take_screenshot_from_url(url: str, output_path: str, wait_time: int = 3) -> str:
    """
    Take a screenshot of a given URL using headless Chrome.
    
    Args:
        url: The URL to screenshot
        output_path: Path to save the screenshot
        wait_time: Seconds to wait for page to render
        headless: Whether to run Chrome in headless mode
        
    Returns:
        Path to the saved screenshot
    """
    print(f"[Screenshot] Capturing: {url}")
    
    options = Options()
    tmp_profile = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={tmp_profile}")
    # options.add_argument("--headless=false")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1024,768")
    
    # Create service with chromedriver
    print(f"[Screenshot] Initializing Chrome driver...")
    if USE_WEBDRIVER_MANAGER:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Fall back to system chromedriver
        driver = webdriver.Chrome(options=options)
    
    try:
        print(f"[Screenshot] Loading page...")
        driver.get(url)
        time.sleep(wait_time)
        driver.save_screenshot(output_path)
        print(f"[Screenshot] Saved to: {output_path}")
    except Exception as e:
        print(f"[Screenshot] Error: {e}")
        raise e
    finally:
        driver.quit()
    
    return output_path


def evaluate_single_task(
    task: dict,
    output_dir: str,
    task_idx: int,
    vlm_model: str,
    fb_model: str,
    enable_webvoyager: bool = False,
    max_tokens: int = 4096
) -> dict:
    """
    Evaluate a single task (prompt + url).
    
    Args:
        task: Dict with 'prompt' and 'url' keys
        output_dir: Directory to save outputs
        task_idx: Index of the task for naming
        vlm_model: Vision-language model for screenshot analysis
        fb_model: Model for feedback generation
        enable_webvoyager: Whether to run WebVoyager GUI agent test
        max_tokens: Max tokens for LLM calls
        
    Returns:
        Evaluation result dict
    """
    start_time = time.time()
    
    prompt = task.get("prompt", "")
    url = task.get("url", "")
    task_id = task.get("id", f"task_{task_idx:04d}")
    
    result = {
        "id": task_id,
        "prompt": prompt,
        "evaluation_prompt": None,
        "url": url,
        "screenshot_path": None,
        "screenshot_analysis": None,
        "screenshot_grade": None,
        "webvoyager_qa_prompt": None,
        "webvoyager_feedback": None,
        "webvoyager_grade": None,
        "status": "success",
        "error_message": None,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": None
    }
    
    task_output_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_output_dir, exist_ok=True)

    try:
        # Step 0: Generate evaluation prompt
        qa_prompt = generate_gui_agent_instruction(prompt, fb_model, max_tokens, max_tokens)
        result["webvoyager_qa_prompt"] = qa_prompt
        # print(f"[QA Prompt] {qa_prompt}")
        if qa_prompt is None:
            raise Exception("Failed to generate QA prompt")
        
        # Step 1: Take screenshot
        screenshot_path = os.path.join(task_output_dir, f"screenshot.png")
        take_screenshot_from_url(url, screenshot_path)
        result["screenshot_path"] = screenshot_path
        
        if not os.path.isfile(screenshot_path):
            raise Exception("Failed to capture screenshot")
        
        # Step 2: Get screenshot description and analysis
        print(f"[Analysis] Analyzing screenshot for task {task_id}...")
        screenshot_analysis = get_screenshot_description(screenshot_path, vlm_model)
        result["screenshot_analysis"] = screenshot_analysis
        
        # Step 3: Get screenshot grade
        print(f"[Grading] Grading screenshot for task {task_id}...")
        grade_result, grade = get_screenshot_grade(screenshot_path, vlm_model, prompt)
        result["screenshot_grade_detail"] = grade_result
        result["screenshot_grade"] = grade
        
        # Step 4: Optional WebVoyager test
        if enable_webvoyager:
            print(f"[WebVoyager] Running GUI agent test for task {task_id}...")
            try:
                webvoyager_text, webvoyager_feedback = get_webvoyager_feedback(
                    idx=task_id,
                    output_dir=task_output_dir,
                    instruction=qa_prompt,
                    url=url,
                    vlm_model=vlm_model,
                    model=fb_model,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_tokens
                )
                result["webvoyager_text"] = webvoyager_text
                result["webvoyager_feedback"] = webvoyager_feedback
                result["webvoyager_grade"] = webvoyager_feedback.get("grade", 0)
            except Exception as e:
                print(f"[WebVoyager] Error: {e}")
                result["webvoyager_error"] = str(e)
        
        print(f"[Done] Task {task_id}: screenshot_grade={grade}, duration={time.time() - start_time:.1f}s")
        
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        print(f"[Error] Task {task_id}: {e}")
    
    # Record duration
    duration = time.time() - start_time
    result["duration_seconds"] = round(duration, 2)
    
    # Save individual result
    result_file = os.path.join(task_output_dir, "result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def compute_summary(results: list, enable_webvoyager: bool) -> dict:
    """Compute summary statistics from results."""
    total = len(results)
    success_results = [r for r in results if r["status"] == "success"]
    success = len(success_results)
    failed = total - success
    
    # Screenshot grades
    screenshot_grades = [r["screenshot_grade"] for r in success_results if r["screenshot_grade"] is not None]
    avg_screenshot_grade = sum(screenshot_grades) / len(screenshot_grades) if screenshot_grades else 0
    
    # Grade distribution
    screenshot_grade_distribution = {}
    for grade in screenshot_grades:
        screenshot_grade_distribution[grade] = screenshot_grade_distribution.get(grade, 0) + 1
    
    summary = {
        "total": total,
        "success": success,
        "failed": failed,
        "avg_screenshot_grade": round(avg_screenshot_grade, 2),
        "screenshot_grade_distribution": screenshot_grade_distribution
    }
    
    # WebVoyager grades if enabled
    if enable_webvoyager:
        webvoyager_grades = [r["webvoyager_grade"] for r in success_results if r.get("webvoyager_grade") is not None]
        avg_webvoyager_grade = sum(webvoyager_grades) / len(webvoyager_grades) if webvoyager_grades else 0
        
        webvoyager_grade_distribution = {}
        for grade in webvoyager_grades:
            webvoyager_grade_distribution[grade] = webvoyager_grade_distribution.get(grade, 0) + 1
        
        summary["avg_webvoyager_grade"] = round(avg_webvoyager_grade, 2)
        summary["webvoyager_grade_distribution"] = webvoyager_grade_distribution
        
        # Pass rate (grade >= 4)
        webvoyager_pass = sum(1 for g in webvoyager_grades if g >= 4)
        summary["webvoyager_pass_rate"] = round(webvoyager_pass / len(webvoyager_grades) * 100, 1) if webvoyager_grades else 0
    
    # Screenshot pass rate (grade >= 3)
    screenshot_pass = sum(1 for g in screenshot_grades if g >= 3)
    summary["screenshot_pass_rate"] = round(screenshot_pass / len(screenshot_grades) * 100, 1) if screenshot_grades else 0
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deployed web URLs using WebGen-Agent feedback capabilities"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with tasks (list of {prompt, url} objects)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file for results (default: {output-dir}/results.json)"
    )
    parser.add_argument(
        "--output-dir", "-d",
        default=None,
        help="Directory to save screenshots and logs (default: OUTPUTS/{timestamp})"
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("VLM_MODEL", "gpt-4o"),
        help="Vision-language model for screenshot analysis"
    )
    parser.add_argument(
        "--fb-model",
        default=os.environ.get("FB_MODEL", "gpt-4o"),
        help="Model for feedback/summary generation"
    )
    parser.add_argument(
        "--enable-webvoyager",
        action="store_true",
        help="Enable WebVoyager GUI agent testing (slower but more comprehensive)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for LLM calls (default: 4096)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for resuming (default: 0)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="End index (exclusive), -1 for all (default: -1)"
    )
    
    args = parser.parse_args()
    
    # Setup default output directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = os.path.join("OUTPUTS", run_timestamp)
    
    # Setup default output file in output directory
    if args.output is None:
        args.output = os.path.join(args.output_dir, "results.json")
    
    # Load input tasks
    print(f"[Init] Loading tasks from: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    
    # Apply index range
    if args.end_idx > 0:
        tasks = tasks[args.start_idx:args.end_idx]
    else:
        tasks = tasks[args.start_idx:]
    
    print(f"[Init] Loaded {len(tasks)} tasks")
    print(f"[Init] VLM Model: {args.vlm_model}")
    print(f"[Init] FB Model: {args.fb_model}")
    print(f"[Init] WebVoyager: {'Enabled' if args.enable_webvoyager else 'Disabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each task
    results = []
    for idx, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[Progress] Task {idx + 1}/{len(tasks)}")
        print(f"{'='*60}")
        
        result = evaluate_single_task(
            task=task,
            output_dir=args.output_dir,
            task_idx=args.start_idx + idx,
            vlm_model=args.vlm_model,
            fb_model=args.fb_model,
            enable_webvoyager=args.enable_webvoyager,
            max_tokens=args.max_tokens
        )
        results.append(result)
        
        # Save intermediate results
        intermediate_output = {
            "results": results,
            "summary": compute_summary(results, args.enable_webvoyager),
            "config": {
                "vlm_model": args.vlm_model,
                "fb_model": args.fb_model,
                "enable_webvoyager": args.enable_webvoyager,
                "timestamp": datetime.now().isoformat()
            }
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(intermediate_output, f, indent=2, ensure_ascii=False)
    
    # Compute final summary
    summary = compute_summary(results, args.enable_webvoyager)
    
    # Final output
    final_output = {
        "results": results,
        "summary": summary,
        "config": {
            "input_file": args.input,
            "vlm_model": args.vlm_model,
            "fb_model": args.fb_model,
            "enable_webvoyager": args.enable_webvoyager,
            "total_tasks": len(tasks),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("[Complete] Evaluation finished!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"Screenshots saved to: {args.output_dir}")
    print(f"\nSummary:")
    print(f"  - Total: {summary['total']}")
    print(f"  - Success: {summary['success']}")
    print(f"  - Failed: {summary['failed']}")
    print(f"  - Avg Screenshot Grade: {summary['avg_screenshot_grade']}/5")
    print(f"  - Screenshot Pass Rate (>=3): {summary['screenshot_pass_rate']}%")
    if args.enable_webvoyager:
        print(f"  - Avg WebVoyager Grade: {summary['avg_webvoyager_grade']}/5")
        print(f"  - WebVoyager Pass Rate (>=4): {summary['webvoyager_pass_rate']}%")


if __name__ == "__main__":
    main()

