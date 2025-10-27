"""
PDF Folder Watcher - Monitors a directory for PDF file changes.
Uses watchdog library for cross-platform file system monitoring.
"""

import os
import time
import threading
from typing import Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class PDFEventHandler(FileSystemEventHandler):
    """Handles PDF file system events with debouncing."""

    def __init__(self, callback: Callable[[str, str], None], debounce_delay: float = 2.0):
        """
        Initialize the event handler.

        Args:
            callback: Function to call on events (signature: callback(event_type, file_path))
            debounce_delay: Delay in seconds to debounce rapid file changes
        """
        super().__init__()
        self.callback = callback
        self.debounce_delay = debounce_delay
        self._pending_events = {}  # {file_path: (event_type, timer)}
        self._lock = threading.Lock()

    def _is_pdf_file(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return file_path.lower().endswith('.pdf')

    def _debounced_callback(self, event_type: str, file_path: str):
        """Execute callback after debounce delay."""
        with self._lock:
            # Clear the pending event
            if file_path in self._pending_events:
                del self._pending_events[file_path]

        # Execute the callback
        try:
            self.callback(event_type, file_path)
        except Exception as e:
            print(f"Error in PDF watcher callback: {str(e)}")

    def _schedule_event(self, event_type: str, file_path: str):
        """Schedule an event with debouncing."""
        with self._lock:
            # Cancel existing timer for this file if any
            if file_path in self._pending_events:
                old_timer = self._pending_events[file_path][1]
                old_timer.cancel()

            # Schedule new timer
            timer = threading.Timer(
                self.debounce_delay,
                self._debounced_callback,
                args=(event_type, file_path)
            )
            self._pending_events[file_path] = (event_type, timer)
            timer.start()

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if not event.is_directory and self._is_pdf_file(event.src_path):
            # Wait a bit to ensure file is fully written
            self._schedule_event('created', event.src_path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if not event.is_directory and self._is_pdf_file(event.src_path):
            self._schedule_event('modified', event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        if not event.is_directory and self._is_pdf_file(event.src_path):
            # Don't debounce deletions - handle immediately
            try:
                self.callback('deleted', event.src_path)
            except Exception as e:
                print(f"Error handling PDF deletion: {str(e)}")


class PDFWatcher:
    """
    Watches a directory for PDF file changes and triggers callbacks.
    Uses watchdog library for efficient file system monitoring.
    """

    def __init__(
        self,
        watch_directory: str,
        callback: Callable[[str, str], None],
        debounce_delay: float = 2.0
    ):
        """
        Initialize the PDF watcher.

        Args:
            watch_directory: Directory to watch for PDF changes
            callback: Function to call on events (signature: callback(event_type, file_path))
            debounce_delay: Delay in seconds to debounce rapid file changes
        """
        self.watch_directory = watch_directory
        self.callback = callback
        self.debounce_delay = debounce_delay

        # Create directory if it doesn't exist
        if not os.path.exists(watch_directory):
            os.makedirs(watch_directory, exist_ok=True)
            print(f"Created watch directory: {watch_directory}")

        # Setup file system observer
        self.event_handler = PDFEventHandler(callback, debounce_delay)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, watch_directory, recursive=False)

        self._is_running = False

    def start(self):
        """Start watching the directory."""
        if not self._is_running:
            self.observer.start()
            self._is_running = True
            print(f"PDF Watcher started - monitoring: {self.watch_directory}")

    def stop(self):
        """Stop watching the directory."""
        if self._is_running:
            self.observer.stop()
            self.observer.join(timeout=5)
            self._is_running = False
            print("PDF Watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._is_running

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
