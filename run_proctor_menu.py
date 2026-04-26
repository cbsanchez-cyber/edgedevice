import os
import subprocess
import sys


DEFAULTS = {
    "mode": "1",
    "camera_source": "0",
    "input_video": "exam.mp4",
    "output_video": "annotated.mp4",
    "fast": "y",
    "headless": "n",
    "event_log_enabled": "y",
    "event_log_path": "event_log.csv",
    "report_csv": "student_report.csv",
    "raspi_mode": "y",
}


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ask_yes_no(prompt: str, default_yes: bool) -> bool:
    default = "y" if default_yes else "n"
    value = input(f"{prompt} (y/n) [{default}]: ").strip().lower()
    if not value:
        value = default
    return value in {"y", "yes", "1", "true"}


def build_command() -> list[str]:
    print("AI Proctor Launcher")
    print("1) Live camera mode")
    print("2) Video upload mode")

    mode = ask("Select mode", DEFAULTS["mode"])
    while mode not in {"1", "2"}:
        mode = ask("Please enter 1 or 2", DEFAULTS["mode"])

    command = [sys.executable, "proctor_edge.py"]

    if mode == "1":
        camera_source = ask("Camera source index", DEFAULTS["camera_source"])
        command.extend(["--source", camera_source])
    else:
        input_video = ask("Input video path", DEFAULTS["input_video"])
        output_video = ask("Annotated output path", DEFAULTS["output_video"])
        command.extend(["--source", input_video, "--output", output_video])

    use_fast = ask_yes_no("Use fast mode", default_yes=True)
    if use_fast:
        command.append("--fast")

    run_headless = ask_yes_no("Run headless", default_yes=False)
    if run_headless:
        command.append("--headless")

    raspi_mode = ask_yes_no("Enable Raspberry Pi mode", default_yes=True)
    if raspi_mode:
        command.append("--raspi")

    event_log_enabled = ask_yes_no("Enable event CSV log", default_yes=True)
    if event_log_enabled:
        event_log_path = ask("Event log CSV path", DEFAULTS["event_log_path"])
        command.extend(["--event-log-csv", event_log_path])
    else:
        command.extend(["--event-log-csv", ""])

    report_csv = ask("Report CSV path", DEFAULTS["report_csv"])
    command.extend(["--report-csv", report_csv])

    return command


def main() -> None:
    command = build_command()
    print("\nFinal command:")
    print(" ".join(command))

    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
