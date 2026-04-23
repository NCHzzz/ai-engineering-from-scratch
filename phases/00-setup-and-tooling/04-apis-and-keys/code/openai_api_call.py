import json
import os
import urllib.request
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def api_base_url() -> str:
    # Quy ước chuẩn:
    # - .env: PROXY_BASE_URL=http://host:port
    # - code tự nối thêm /v1
    #
    # Nếu bạn muốn để PROXY_BASE_URL đã có /v1 trong .env,
    # thì đổi hàm này thành: return require_env("PROXY_BASE_URL").rstrip("/")
    return require_env("PROXY_BASE_URL").rstrip("/") + "/v1"


def call_with_sdk_streaming() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install the SDK: uv pip install openai python-dotenv")
        return

    token = require_env("PROXY_TOKEN")
    model = require_env("PROXY_MODEL")
    base_url = api_base_url()

    client = OpenAI(
        api_key=token,
        base_url=base_url,
        timeout=60.0,
    )

    print(f"SDK URL: {base_url}")
    print("SDK response: ", end="", flush=True)

    final_response = None

    try:
        # Responses API streaming
        with client.responses.stream(
            model=model,
            input="What is a neural network in one sentence?",
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.error":
                    # Hiển thị lỗi nếu proxy trả event lỗi
                    print(f"\n[stream error] {event.error}", flush=True)

            final_response = stream.get_final_response()

        print()

        if final_response and getattr(final_response, "usage", None):
            print(
                f"Tokens used: "
                f"{final_response.usage.input_tokens} in, "
                f"{final_response.usage.output_tokens} out"
            )

    except Exception as e:
        print(f"\nSDK error: {repr(e)}")


def call_raw_http_streaming() -> None:
    token = require_env("PROXY_TOKEN")
    model = require_env("PROXY_MODEL")
    url = api_base_url().rstrip("/") + "/responses"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    body = json.dumps(
        {
            "model": model,
            "input": "What is a neural network in one sentence?",
            "stream": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    print(f"RAW URL: {url}")
    print("Raw HTTP response: ", end="", flush=True)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            completed_usage = None

            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()

                # SSE keep-alive / dòng trống
                if not line or line.startswith(":"):
                    continue

                # SSE data line
                if not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()
                if not payload:
                    continue

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    # Bỏ qua event không parse được
                    continue

                event_type = event.get("type")

                if event_type == "response.output_text.delta":
                    print(event.get("delta", ""), end="", flush=True)

                elif event_type == "response.error":
                    error_obj = event.get("error", {})
                    message = error_obj.get("message", "Unknown streaming error")
                    raise RuntimeError(message)

                elif event_type == "response.completed":
                    response_obj = event.get("response", {})
                    completed_usage = response_obj.get("usage")

            print()

            if completed_usage:
                print(
                    f"Tokens used: "
                    f"{completed_usage.get('input_tokens', 0)} in, "
                    f"{completed_usage.get('output_tokens', 0)} out"
                )

    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"\nHTTP error: {e.code} {e.reason}")
        print(body)
    except URLError as e:
        print(f"\nConnection error: {e.reason}")
    except Exception as e:
        print(f"\nRaw HTTP error: {repr(e)}")


if __name__ == "__main__":
    print("=== API Calls (Streaming) ===\n")

    print("1. Using the SDK:")
    call_with_sdk_streaming()

    print("\n2. Using raw HTTP:")
    call_raw_http_streaming()