"""Launch Streamlit and retain startup diagnostics for the installed app."""
import os
from pathlib import Path
import socket
import sys
import traceback


if __name__ == "__main__":
    log_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CPM Model Builder" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = (log_root / "startup.log").open("a", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        from streamlit.web import cli
        app_file = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "app.py"
        port = os.environ.get("CPM_SERVER_PORT")
        if not port:
            with socket.socket() as port_probe:
                port_probe.bind(("127.0.0.1", 0))
                port = str(port_probe.getsockname()[1])
        sys.argv = ["streamlit", "run", str(app_file), "--server.address=127.0.0.1",
                    f"--server.port={port}", "--global.developmentMode=false"]
        raise SystemExit(cli.main())
    except Exception:
        traceback.print_exc(file=log_file)
        log_file.flush()
        raise
    finally:
        log_file.flush()
        log_file.close()
