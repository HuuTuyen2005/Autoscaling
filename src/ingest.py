# src/ingest.py
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

LOG_PATTERN = re.compile(
    r'(?P<host>\S+) '
    r'\S+ \S+ '
    r'\[(?P<time>.*?)\] '
    r'"(?P<method>\S+) (?P<url>\S+) \S+" '
    r'(?P<status>\d{3}) '
    r'(?P<bytes>\S+)'
)

def parse_log_line(line: str):
    match = LOG_PATTERN.search(line)
    if not match:
        return None

    data = match.groupdict()

    timestamp = datetime.strptime(
        data["time"], "%d/%b/%Y:%H:%M:%S %z"
    )

    bytes_sent = data["bytes"]
    bytes_sent = int(bytes_sent) if bytes_sent != "-" else 0

    return {
        "timestamp": timestamp,
        "host": data["host"],
        "method": data["method"],
        "url": data["url"],
        "status": int(data["status"]),
        "bytes": bytes_sent
    }


def load_log_file(file_path: str) -> pd.DataFrame:
    rows = []
    with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                rows.append(parsed)

    df = pd.DataFrame(rows)
    return df


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        file_path = raw_dir / f"{split}.txt"
        print(f"Parsing {file_path} ...")

        df = load_log_file(file_path)

        df = df.sort_values("timestamp")

        out_path = processed_dir / f"{split}_parsed.csv"
        df.to_csv(out_path, index=False)


        print(f"Saved parsed data to {out_path}")
        print(df.head())


if __name__ == "__main__":
    main()
