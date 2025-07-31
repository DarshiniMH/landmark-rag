import os, pathlib, random

SEED_FILE = pathlib.Path(".eval_seed")

def get_run_seed() -> int:
    # 1. Honor EVAL_SEED if set (for CI overrides)
    if os.getenv("EVAL_SEED"):
        return int(os.getenv("EVAL_SEED"))

    # 2. Otherwise, if .eval_seed exists, reuse it
    if SEED_FILE.exists():
        return int(SEED_FILE.read_text().strip())

    # 3. Else, generate one, persist it, and return it
    seed = random.randint(1, 1_000_000)
    SEED_FILE.write_text(str(seed))
    return seed
