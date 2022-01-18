"""
Data Logging Module
"""

from typing import Union, Optional
from os.path import join
from os import makedirs
import logging
from uuid import uuid4

import sqlite3
from cv2 import cv2
import numpy as np

class DataLogging():
    """
    Data Logging Class
    """
    def __init__(self, path: Optional[str] = None, table_name: Optional[str] = None):
        self.database = None
        self.cursor = None
        self.data_id = None
        self.database_name = 'log.db'
        self.table_name = table_name or 'logs'
        self.path = path or './logs'
        self.image_path = 'images'
        self._init_log_directory()
        self._init_database()
        self.set_new_id()

    def _init_database(self):
        self.database = self._get_database()
        if self.database is not None:
            self.cursor = self.database.cursor()
            self.cursor.execute(
                f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT,
                    architecture_name TEXT,
                    detection_result TEXT,
                    confidence DOUBLE,
                    pos_x INTEGER,
                    pos_y INTEGER,
                    pos_w INTEGER,
                    pos_h INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')

    def _init_log_directory(self):
        try:
            makedirs(join(self.path, self.image_path))
        except OSError:
            return

    def set_new_id(self):
        """
        Change current id to a new UUID
        """
        self.data_id = uuid4().hex

    def _get_database(self) -> Optional[sqlite3.Connection] :
        try:
            database = sqlite3.connect(join(self.path, self.database_name))
        except sqlite3.Error:
            logging.error('Database could not be initialised.')
            return None
        return database

    def log_image(self, image: np.ndarray):
        """
        Log Image to File
        """
        cv2.imwrite(join(self.path, self.image_path, self.data_id + '.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 70])

    def log_metrics(self, data: dict[str, Union[str, int, float, None]]):
        """
        Log Textual Metric Data
        """
        if self.cursor is None:
            return
        self.cursor.execute(f'INSERT INTO {self.table_name} (id, detection_result, confidence, architecture_name, pos_x, pos_y, pos_w, pos_h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [
            self.data_id,
            data['detection_result'],
            data['confidence'],
            data['architecture_name'],
            data['pos_x'],
            data['pos_y'],
            data['pos_w'],
            data['pos_h']
        ])
        self.database.commit()
