"""
conftest.py — общие pytest-фикстуры для тестов inmoov_voice.

Запуск тестов:
  colcon test --packages-select inmoov_voice      # все тесты
  colcon test --packages-select inmoov_voice --pytest-args -m units
  colcon test --packages-select inmoov_voice --pytest-args -m services
  colcon test --packages-select inmoov_voice --pytest-args -m nodes
  colcon test --packages-select inmoov_voice --pytest-args -v

Или напрямую через pytest (из ros2_ws, после source install/setup.bash):
  pytest src/inmoov_voice/test/ -v
  pytest src/inmoov_voice/test/ -m "units"
  pytest src/inmoov_voice/test/ -m "services" -v
  pytest src/inmoov_voice/test/ -m "nodes" --timeout=30
"""

import pytest
import rclpy


def pytest_configure(config):
    config.addinivalue_line('markers', 'units:    быстрые unit-тесты без ROS2 и сети')
    config.addinivalue_line('markers', 'services: проверка доступности внешних сервисов')
    config.addinivalue_line('markers', 'nodes:    тесты запуска ROS2 нод (мокают HTTP)')
    config.addinivalue_line('markers', 'pipeline: end-to-end тесты (нужны сервисы)')
    config.addinivalue_line('markers', 'slow:     медленные тесты (загрузка моделей)')
    config.addinivalue_line('markers', 'hardware: требуют физического железа')


@pytest.fixture(scope='session')
def ros():
    """Инициализирует rclpy один раз на всю pytest-сессию."""
    if not rclpy.ok():
        rclpy.init()
    yield
    # Не делаем rclpy.shutdown() — colcon test запускает каждый пакет отдельно.
