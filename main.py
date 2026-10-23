"""Launch the Medical Image PDE Segmentation Dashboard.

This is the original entry point — ``python main.py`` still works
exactly as documented.  All logic lives in the ``levelset_seg`` package.
"""

from levelset_seg.gui import BrainTumorMasterApp

if __name__ == "__main__":
    app = BrainTumorMasterApp()
    app.mainloop()