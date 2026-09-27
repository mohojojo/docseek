from __future__ import annotations

import base64
from unittest.mock import MagicMock

from docseek.scraper import capture_screenshot


def test_capture_screenshot_no_scaling():
    """Happy path: viewport already under max_width, no scaling needed."""
    page = MagicMock()
    page.viewport_size = {'width': 800, 'height': 600}

    # Mock screenshot to return test PNG bytes
    test_png = b'\x89PNG\x00\x01\x02\x03'
    page.screenshot.return_value = test_png

    result = capture_screenshot(page, max_width=800)

    # Verify result is non-empty string that base64-decodes to original bytes
    assert result is not None
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded == test_png

    # Verify set_viewport_size was NOT called (no resizing)
    page.set_viewport_size.assert_not_called()


def test_capture_screenshot_with_scaling():
    """Scaling: viewport exceeds max_width, should resize, capture, and restore."""
    page = MagicMock()
    page.viewport_size = {'width': 1920, 'height': 1080}

    # Mock screenshot
    test_png = b'\x89PNG\x04\x05\x06\x07'
    page.screenshot.return_value = test_png

    result = capture_screenshot(page, max_width=800)

    # Verify result is valid base64
    assert result is not None
    decoded = base64.b64decode(result)
    assert decoded == test_png

    # Verify set_viewport_size was called twice:
    # 1. To resize to max_width (with scaled height)
    # 2. To restore original viewport
    assert page.set_viewport_size.call_count == 2

    # Check the resize call (first call)
    first_call = page.set_viewport_size.call_args_list[0]
    expected_height = round(1080 * 800 / 1920)  # should be 450
    assert first_call[0][0] == {'width': 800, 'height': expected_height}

    # Check the restore call (second call)
    second_call = page.set_viewport_size.call_args_list[1]
    assert second_call[0][0] == {'width': 1920, 'height': 1080}


def test_capture_screenshot_exception_returns_none():
    """Failure: screenshot() raises exception, function returns None without propagating."""
    page = MagicMock()
    page.viewport_size = {'width': 800, 'height': 600}

    # Mock screenshot to raise exception
    page.screenshot.side_effect = Exception('browser closed')

    result = capture_screenshot(page, max_width=800)

    # Verify function returns None (not raising)
    assert result is None


def test_capture_screenshot_exception_after_resize():
    """Failure after resize: screenshot() raises, set_viewport_size is called twice (resize + restore), returns None."""
    page = MagicMock()
    page.viewport_size = {'width': 1920, 'height': 1080}

    # Mock screenshot to raise exception after viewport is resized
    page.screenshot.side_effect = Exception('browser closed')

    result = capture_screenshot(page, max_width=800)

    # Verify function returns None (not raising)
    assert result is None

    # Verify set_viewport_size was called twice:
    # 1. To resize to max_width (with scaled height)
    # 2. To restore original viewport in the finally block
    assert page.set_viewport_size.call_count == 2

    # Check the resize call (first call)
    first_call = page.set_viewport_size.call_args_list[0]
    expected_height = round(1080 * 800 / 1920)  # should be 450
    assert first_call[0][0] == {'width': 800, 'height': expected_height}

    # Check the restore call (second call)
    second_call = page.set_viewport_size.call_args_list[1]
    assert second_call[0][0] == {'width': 1920, 'height': 1080}


# ---------------------------------------------------------------------------
# select_option_anywhere: <select> and ARIA comboboxes
# ---------------------------------------------------------------------------

class TestSelectOptionAnywhere:
    """Component libraries render a button plus a hidden listbox instead of a <select>.

    Playwright's select_option waits for an element that never becomes a select, so it times out.
    One bank's monthly reports sat behind exactly such a year filter.
    """

    def _page(self, tag: str, count: int = 1, pick=None):
        page = MagicMock()
        element = MagicMock()
        element.count.return_value = count
        element.evaluate.return_value = tag
        page.locator.return_value = element
        page.evaluate.return_value = pick if pick is not None else {'ok': True, 'text': '2026'}
        return page, element

    def test_native_select_uses_playwright_select_option(self):
        from docseek.scraper import select_option_anywhere
        page, element = self._page('SELECT')
        result = select_option_anywhere(page, '7', 'Hungary')
        element.select_option.assert_called_once()
        element.click.assert_not_called()
        assert 'Hungary' in result

    def test_native_select_falls_back_from_label_to_value(self):
        from docseek.scraper import select_option_anywhere
        page, element = self._page('SELECT')
        element.select_option.side_effect = [Exception('no such label'), None]
        select_option_anywhere(page, '7', 'HU')
        assert element.select_option.call_count == 2

    def test_aria_combobox_is_opened_then_the_option_clicked(self):
        from docseek.scraper import select_option_anywhere
        page, element = self._page('BUTTON')
        result = select_option_anywhere(page, '94', '2026')
        element.click.assert_called_once()          # opens the listbox
        element.select_option.assert_not_called()   # would time out on a button
        page.evaluate.assert_called_once()          # picks the option
        assert '2026' in result

    def test_a_missing_option_reports_what_was_offered(self):
        from docseek.scraper import select_option_anywhere
        page, _ = self._page('DIV', pick={'ok': False, 'options': ['2026', '2025', '2024']})
        result = select_option_anywhere(page, '94', '1999')
        assert '1999' in result and '2026' in result   # the agent can retry with a real option

    def test_a_missing_element_is_reported_not_raised(self):
        from docseek.scraper import select_option_anywhere
        page, _ = self._page('SELECT', count=0)
        assert 'not found' in select_option_anywhere(page, '404', 'x')
