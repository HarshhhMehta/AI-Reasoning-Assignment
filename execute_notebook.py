import nbformat
import sys
import io
import contextlib
import os

notebook_path = os.path.join("submission", "reasoning_agent.ipynb")
print(f"Loading notebook: {notebook_path}")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Create an execution namespace
namespace = {
    "__file__": os.path.abspath(notebook_path),
    "__name__": "__main__",
}

print(f"Executing {len(nb.cells)} cells...")

execution_count = 1
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        code = cell.source
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        print(f"  Executing cell {execution_count}...", end=" ", flush=True)
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, namespace)
            
            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()
            
            outputs = []
            if out_text:
                outputs.append(nbformat.v4.new_output(output_type="stream", name="stdout", text=out_text))
            if err_text:
                outputs.append(nbformat.v4.new_output(output_type="stream", name="stderr", text=err_text))
            
            cell.outputs = outputs
            cell.execution_count = execution_count
            execution_count += 1
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            tb = traceback.format_exc()
            cell.outputs = [
                nbformat.v4.new_output(output_type="error", ename=type(e).__name__, evalue=str(e), traceback=tb.splitlines())
            ]
            cell.execution_count = execution_count
            execution_count += 1
            raise e

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook successfully executed and saved with outputs to: {notebook_path}")
