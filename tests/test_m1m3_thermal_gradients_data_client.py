# This file is part of ts_ess_m1m3.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import asyncio
import logging
import types
import unittest

try:
    from lsst.ts.ess.common.data_client import get_data_client_class
    from lsst.ts.ess.m1m3 import ThermalGradientsDataClient
except ImportError:
    ThermalGradientsDataClient = None

# Time limit (sec) to receive the first published gradients.
PUBLISH_TIMEOUT = 10


class MockWriteTopic:
    """Minimal stand-in for a salobj write topic."""

    def __init__(self) -> None:
        self.data: list[dict] = []
        self.written = asyncio.Event()

    async def set_write(self, **kwargs: float) -> None:
        self.data.append(kwargs)
        self.written.set()


@unittest.skipIf(ThermalGradientsDataClient is None, "ts_ess_common is not available")
class M1M3ThermalGradientsDataClientTestCase(unittest.IsolatedAsyncioTestCase):
    def make_config(self, **overrides: object) -> types.SimpleNamespace:
        """Config matching the get_config_schema defaults."""
        schema = ThermalGradientsDataClient.get_config_schema()
        config = {name: properties["default"] for name, properties in schema["properties"].items()}
        config.update(overrides)
        return types.SimpleNamespace(**config)

    async def test_registry(self) -> None:
        self.assertIs(
            get_data_client_class("ThermalGradientsDataClient"),
            ThermalGradientsDataClient,
        )

    async def test_publishes_simulated_gradients(self) -> None:
        topic = MockWriteTopic()
        topics = types.SimpleNamespace(tel_m1m3ThermalGradients=topic)
        client = ThermalGradientsDataClient(
            config=self.make_config(remove_nonstandard_cells=False),
            topics=topics,
            log=logging.getLogger(),
            simulation_mode=1,
        )

        async with client:
            async with asyncio.timeout(PUBLISH_TIMEOUT):
                await topic.written.wait()

        self.assertGreaterEqual(len(topic.data), 1)
        gradients = topic.data[0]
        self.assertEqual(
            set(gradients),
            {
                "xGradient",
                "yGradient",
                "zGradient",
                "radialGradient",
                "xGradientError",
                "yGradientError",
                "zGradientError",
                "radialGradientError",
            },
        )
        self.assertAlmostEqual(
            gradients["xGradient"],
            ThermalGradientsDataClient.SIMULATED_X_GRADIENT,
            places=6,
        )
        self.assertAlmostEqual(
            gradients["yGradient"],
            ThermalGradientsDataClient.SIMULATED_Y_GRADIENT,
            places=6,
        )
        self.assertAlmostEqual(
            gradients["zGradient"],
            ThermalGradientsDataClient.SIMULATED_Z_GRADIENT,
            places=6,
        )
        self.assertAlmostEqual(gradients["xGradientError"], 0, places=6)

    async def test_min_publish_interval(self) -> None:
        topic = MockWriteTopic()
        topics = types.SimpleNamespace(tel_m1m3ThermalGradients=topic)
        # With a large min_publish_interval only one sample is published,
        # no matter how many simulated scans complete.
        client = ThermalGradientsDataClient(
            config=self.make_config(remove_nonstandard_cells=False, min_publish_interval=3600),
            topics=topics,
            log=logging.getLogger(),
            simulation_mode=1,
        )

        async with client:
            async with asyncio.timeout(PUBLISH_TIMEOUT):
                await topic.written.wait()
            await asyncio.sleep(5)

        self.assertEqual(len(topic.data), 1)


if __name__ == "__main__":
    unittest.main()
