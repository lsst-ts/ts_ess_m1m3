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

from __future__ import annotations

__all__ = ["ThermalGradientsDataClient"]

import asyncio
import functools
import logging
import math
import types
import typing

import yaml

from lsst.ts import utils
from lsst.ts.ess.common.data_client import BaseReadLoopDataClient
from lsst.ts.m1m3.utils import ThermocoupleCache, fit_thermal_gradients, thermocouple_z_position
from lsst.ts.xml.tables.m1m3 import (
    Scanner,
    find_thermocouple,
    set_air_nozzles_types_and_orifice_diameters,
)

# Number of ESS.temperature messages each thermal scanner publishes
# per scan: ceil(95 thermocouple channels / 16 channels per message).
MESSAGES_PER_SCANNER = 6

# Interval (sec) between simulated scans in simulation mode.
SIMULATION_INTERVAL = 2.0


class ThermalGradientsDataClient(BaseReadLoopDataClient):
    """Compute M1M3 glass thermal gradients from the thermal scanner
    temperatures and publish them as ESS.m1m3ThermalGradients telemetry.

    Subscribes to the ESS.temperature telemetry published by the four
    M1M3 GEC thermal scanner instances (SAL indices 114-117), caches the
    per-thermocouple temperatures in a `ThermocoupleCache` and - whenever
    a valid (complete and fresh) set of thermocouple values is available -
    fits thermal gradients with `fit_thermal_gradients` and writes the
    ESS.m1m3ThermalGradients topic.

    Also subscribes to the MTM1M3TS.airNozzles event to keep the air
    nozzle table current, so thermocouples in cells with nonstandard air
    nozzle configurations can be excluded from the fit.

    Parameters
    ----------
    config : `types.SimpleNamespace`
        The configuration, after validation by the schema returned
        by `get_config_schema` and conversion to a types.SimpleNamespace.
    topics : `salobj.Controller`
        The telemetry topics this data client can write, as a struct
        with attribute ``tel_m1m3ThermalGradients``. In normal operation
        this is the ESS CSC, which also provides the ``domain`` used to
        create the read-only remotes.
    log : `logging.Logger`
        Logger.
    simulation_mode : `int`, optional
        Simulation mode; 0 for normal operation. In simulation mode no
        remotes are created; scanner telemetry with a known temperature
        plane (see the SIMULATED_* class attributes) is generated
        internally.
    """

    # Coefficients of the temperature plane generated in simulation mode.
    SIMULATED_INTERCEPT = 10.0
    SIMULATED_X_GRADIENT = 0.5
    SIMULATED_Y_GRADIENT = -0.3
    SIMULATED_Z_GRADIENT = 1.2

    def __init__(
        self,
        config: types.SimpleNamespace,
        topics: salobj.Controller | types.SimpleNamespace,
        log: logging.Logger,
        simulation_mode: int = 0,
    ) -> None:
        super().__init__(
            config=config,
            topics=topics,
            log=log,
            simulation_mode=simulation_mode,
        )

        self.cache = ThermocoupleCache(
            max_data_age=self.config.max_data_age,
            max_missing=self.config.max_missing,
        )
        self.last_published_timestamp = -math.inf

        self._queue: asyncio.Queue[tuple[int, typing.Any]] = asyncio.Queue()
        self._remotes: list[salobj.Remote] = []
        self._simulation_task = utils.make_done_future()

    @classmethod
    def get_config_schema(cls) -> dict[str, typing.Any]:
        return yaml.safe_load(
            """
$schema: http://json-schema.org/draft-07/schema#
description: Schema for MTM1M3 ThermalGradientsDataClient
type: object
properties:
  max_data_age:
    description: >-
      Maximum age (sec) of a cached thermocouple sample, relative to the
      newest cached sample, for it to count towards a valid set.
    type: number
    default: 120
  max_missing:
    description: >-
      Maximum number of thermocouples that can be missing or stale while
      the set is still considered valid and gradients are published.
    type: integer
    default: 0
  min_publish_interval:
    description: >-
      Minimum interval (sec, in scanner timestamps) between published
      gradient samples.
    type: number
    default: 30
  radius_limit:
    description: >-
      Only use thermocouples within this radius (m) from the mirror
      center. Null means use all thermocouples.
    type:
    - number
    - "null"
    default: null
  remove_nonstandard_cells:
    description: >-
      Exclude thermocouples in cells with nonstandard air nozzle
      configurations (as reported by the MTM1M3TS airNozzles event)
      from the gradient fit.
    type: boolean
    default: true
  connect_timeout:
    description: Timeout for starting the ESS and MTM1M3TS remotes (sec).
    type: number
    default: 60
  read_timeout:
    description: >-
      Timeout for receiving the next ESS.temperature message (sec).
      Should comfortably exceed the 30 sec scanner cadence.
    type: number
    default: 120
  max_read_timeouts:
    description: Maximum number of read timeouts before an exception is raised.
    type: integer
    default: 5
required:
  - max_data_age
  - max_missing
  - min_publish_interval
  - radius_limit
  - remove_nonstandard_cells
  - connect_timeout
  - read_timeout
  - max_read_timeouts
additionalProperties: false
"""
        )

    def descr(self) -> str:
        return f"scanners {[int(scanner) for scanner in Scanner]} -> ESS.m1m3ThermalGradients"

    async def connect(self) -> None:
        if self.simulation_mode > 0:
            self._simulation_task = asyncio.create_task(self._simulation_loop())
        else:
            # Import here so this module can be used (e.g. in simulation
            # mode) without a full salobj installation.
            from lsst.ts import salobj

            domain = getattr(self.topics, "domain", None)
            if domain is None:
                raise RuntimeError("topics has no domain; cannot create remotes for the thermal scanners")

            for scanner in Scanner:
                remote = salobj.Remote(
                    domain=domain,
                    name="ESS",
                    index=int(scanner),
                    include=["temperature"],
                    start=False,
                )
                remote.tel_temperature.callback = functools.partial(self._handle_temperature, int(scanner))
                self._remotes.append(remote)

            if self.config.remove_nonstandard_cells:
                remote = salobj.Remote(
                    domain=domain,
                    name="MTM1M3TS",
                    include=["airNozzles"],
                    start=False,
                )
                remote.evt_airNozzles.callback = self._handle_air_nozzles
                self._remotes.append(remote)

            async with asyncio.timeout(self.config.connect_timeout):
                await asyncio.gather(*[remote.start() for remote in self._remotes])

        await super().connect()

    async def disconnect(self) -> None:
        self._simulation_task.cancel()
        try:
            for remote in self._remotes:
                await remote.close()
        finally:
            self._remotes = []
            await super().disconnect()

    async def _handle_temperature(self, sal_index: int, data: typing.Any) -> None:
        self._queue.put_nowait((sal_index, data))

    async def _handle_air_nozzles(self, data: typing.Any) -> None:
        self.log.info("Updating air nozzle table from MTM1M3TS.airNozzles event.")
        set_air_nozzles_types_and_orifice_diameters(data)

    async def read_data(self) -> None:
        """Process one ESS.temperature message; publish gradients when a
        valid set of thermocouple temperatures is completed.
        """
        async with asyncio.timeout(self.read_timeout):
            messages = [await self._queue.get()]
        # Scanners publish their channels as a burst of messages;
        # process everything that has arrived.
        while not self._queue.empty():
            messages.append(self._queue.get_nowait())

        updated = False
        for sal_index, data in messages:
            try:
                updated |= self.cache.add_temperatures(
                    sal_index=sal_index,
                    sensor_name=data.sensorName,
                    timestamp=data.timestamp,
                    temperatures=data.temperatureItem,
                )
            except ValueError as error:
                self.log.warning(f"Ignoring unexpected temperature message: {error}")

        if not updated:
            return

        temperatures = self.cache.valid_set()
        if temperatures is None:
            self.log.debug(
                f"No valid thermocouple set yet; {len(self.cache.missing_names())} missing or stale."
            )
            return

        newest = self.cache.newest_timestamp
        if newest - self.last_published_timestamp < self.config.min_publish_interval:
            return

        gradients = fit_thermal_gradients(
            temperatures,
            remove_nonstandard_cells=self.config.remove_nonstandard_cells,
            radius_limit=self.config.radius_limit,
        )

        await self.topics.tel_m1m3ThermalGradients.set_write(
            xGradient=gradients.x_gradient,
            yGradient=gradients.y_gradient,
            zGradient=gradients.z_gradient,
            radialGradient=gradients.radial_gradient,
            xGradientError=gradients.x_gradient_err,
            yGradientError=gradients.y_gradient_err,
            zGradientError=gradients.z_gradient_err,
            radialGradientError=gradients.radial_gradient_err,
        )
        self.last_published_timestamp = newest

    def _simulated_temperature(self, scanner: Scanner, channel: int) -> float:
        """Return the simulated temperature for one scanner channel."""
        thermocouple = find_thermocouple(scanner, channel)
        if thermocouple is None:
            return 0.0
        return (
            self.SIMULATED_INTERCEPT
            + self.SIMULATED_X_GRADIENT * thermocouple.x_position
            + self.SIMULATED_Y_GRADIENT * thermocouple.y_position
            + self.SIMULATED_Z_GRADIENT * thermocouple_z_position(thermocouple.name)
        )

    async def _simulation_loop(self) -> None:
        """Generate scanner telemetry sampling a known temperature plane."""
        while True:
            timestamp = utils.current_tai()
            for scanner in Scanner:
                for chunk_index in range(MESSAGES_PER_SCANNER):
                    temperatures = [
                        self._simulated_temperature(
                            scanner, chunk_index * ThermocoupleCache.CHANNELS_PER_MESSAGE + channel
                        )
                        for channel in range(ThermocoupleCache.CHANNELS_PER_MESSAGE)
                    ]
                    data = types.SimpleNamespace(
                        sensorName=f"m1m3-ts-{int(scanner) - 113:02d} "
                        f"{chunk_index + 1}/{MESSAGES_PER_SCANNER}",
                        timestamp=timestamp,
                        temperatureItem=temperatures,
                    )
                    self._queue.put_nowait((int(scanner), data))
            await asyncio.sleep(SIMULATION_INTERVAL)
