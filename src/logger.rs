/*
 * Copyright© 2025 UT-Battelle, LLC
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
use flexi_logger::{FileSpec, LoggerHandle, Logger, WriteMode};


/// Initialize the logger
pub fn init_logger() -> LoggerHandle
{
    let logger = Logger::try_with_str("info").unwrap()
        .log_to_file(
            FileSpec::default()
            .basename("ormatex_rs")
            .suppress_timestamp()
            .suffix("log"))
        .write_mode(WriteMode::BufferAndFlush)
        .start().unwrap();
    log::info!("ORMATEX Log");
    logger
}

