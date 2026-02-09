from pathlib import Path

from app.agent.admin.workflow import build_graph


def test_build_graph():
    png_bytes = build_graph().get_graph().draw_mermaid_png()
    assert png_bytes

    try:
        from IPython import get_ipython
        from IPython.display import Image, display

        if get_ipython() is not None:
            display(Image(png_bytes))
            return
    except Exception:
        pass

    output_path = Path("tmp/admin_workflow_graph.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    assert output_path.exists()
    print(f"graph image saved to: {output_path.resolve()}")
