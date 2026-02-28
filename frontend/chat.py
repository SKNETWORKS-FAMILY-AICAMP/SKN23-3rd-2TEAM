# -*- coding: utf-8 -*-
"""Chat page bridge.

The newly introduced `frontend/chat.py` is wired to the existing production
chat implementation in `frontend/chat_ui.py` to preserve behavior.
"""

from frontend.chat_ui import show_chat_page as _show_chat_page


def show_chat_page():
    _show_chat_page()

