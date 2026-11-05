"""Path and fresh-output checks shared by qualification."""
from pathlib import Path


def _fresh(output,inputs):
    output=Path(output).resolve()
    for item in inputs:
        item=Path(item).resolve()
        if output.is_relative_to(item) or item.is_relative_to(output):
            raise ValueError('Keep guarded outputs separate from all input/reference directories and files')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new/empty guarded output directory')
    return output


def _locked_new_output(output):
    if any(p.name!='.batch.lock' for p in Path(output).iterdir()):
        raise FileExistsError('Output acquired new content before lock acquisition; preserve existing evidence')
