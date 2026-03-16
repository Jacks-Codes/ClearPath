#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import subprocess
from pathlib import Path

# Paths to check
package_path = Path(__file__).parents[2] / "ibm_watsonx_ai"
docs_path = Path(__file__).parents[2] / "docs"

cmd = ["typos", str(package_path), str(docs_path)]

subprocess.run(cmd)
