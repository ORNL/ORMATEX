/*
 * Copyright(c) 2025 UT-Battelle, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//! File logging setup for ORMATEX.
//!
//! Provides [`init_logger`], which starts a `flexi_logger` file logger that
//! collects the `log` crate messages emitted by the integrators. It is enabled
//! from python through the `logging=True` keyword of the rust integrate wrapper.
use flexi_logger::{FileSpec, Logger, LoggerHandle, WriteMode};

/// Initialize the file logger.
///
/// Starts a logger at level `info` that writes to the file `ormatex_rs.log`
/// (no timestamp in the file name) in the default `flexi_logger` output
/// directory, which is the current working directory. Writes are buffered and
/// flushed periodically. The first message logged is `ORMATEX Log`.
///
/// # Returns
///
/// The [`LoggerHandle`]. Keep it alive for as long as logging is needed;
/// dropping it shuts the logger down.
///
/// # Panics
///
/// Panics if the logger configuration is invalid or the logger cannot be
/// started, for example if a global logger is already installed or the log file
/// cannot be created.
pub fn init_logger() -> LoggerHandle {
    let logger = Logger::try_with_str("info")
        .unwrap()
        .log_to_file(
            FileSpec::default()
                .basename("ormatex_rs")
                .suppress_timestamp()
                .suffix("log"),
        )
        .write_mode(WriteMode::BufferAndFlush)
        .start()
        .unwrap();
    log::info!("ORMATEX Log");
    logger
}
