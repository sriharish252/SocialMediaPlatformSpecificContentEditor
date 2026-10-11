import sys
from pathlib import Path

from streamlit.web import cli


def main() -> None:
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
