suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
config_path <- "Libs/configs/runtime/osp_pksim_environment.yaml"
model_path <- NA_character_
output_path <- NA_character_
standardize_csv_input <- NA_character_
dose_parameter_path <- NA_character_
dose_value <- NA_real_
dose_unit <- NA_character_
plasma_regex <- "plasma|venous|arterial"
csf_regex <- "csf|cerebrospinal"
brain_isf_regex <- "brain|interstitial|isf|extracellular|intracellular"
result_file_regex <- ".*-Results\\.csv$"
csv_metadata_columns <- character()
measure_column_keep_regex <- NA_character_
standardization_policy_uid <- NA_character_
max_measure_columns_per_chunk <- NA_integer_
if (length(args) %% 2 != 0) {
  stop("Arguments must be provided as --key value pairs")
}
if (length(args) > 0) {
  for (index in seq(1, length(args), by = 2)) {
    key <- args[[index]]
    value <- args[[index + 1]]
    if (key == "--config") config_path <- value
    if (key == "--model") model_path <- value
    if (key == "--output") output_path <- value
    if (key == "--standardize-csv-input") standardize_csv_input <- value
    if (key == "--dose-parameter") dose_parameter_path <- value
    if (key == "--dose") dose_value <- as.numeric(value)
    if (key == "--dose-unit") dose_unit <- value
    if (key == "--plasma-regex") plasma_regex <- value
    if (key == "--csf-regex") csf_regex <- value
    if (key == "--brain-isf-regex") brain_isf_regex <- value
    if (key == "--result-file-regex") result_file_regex <- value
    if (key == "--csv-metadata-columns") csv_metadata_columns <- strsplit(value, "\\|", fixed = FALSE)[[1]]
    if (key == "--measure-column-keep-regex") measure_column_keep_regex <- value
    if (key == "--standardization-policy-uid") standardization_policy_uid <- value
    if (key == "--max-measure-columns-per-chunk") max_measure_columns_per_chunk <- as.integer(value)
  }
}
if (is.na(output_path)) stop("--output <tsv> is required")
if (is.na(max_measure_columns_per_chunk) || max_measure_columns_per_chunk <= 0) {
  stop("--max-measure-columns-per-chunk must be a positive integer")
}
if (length(csv_metadata_columns) == 0) {
  stop("--csv-metadata-columns must declare OSPSuite non-measure CSV metadata columns")
}
if (is.na(measure_column_keep_regex) || nchar(measure_column_keep_regex) == 0) {
  stop("--measure-column-keep-regex must declare the OSPSuite output paths allowed for PBPK teacher standardization")
}
if (is.na(standardization_policy_uid) || nchar(standardization_policy_uid) == 0) {
  stop("--standardization-policy-uid must declare the OSPSuite standardization policy used for cache validation")
}
if (!is.na(standardize_csv_input)) {
  if (!file.exists(standardize_csv_input)) stop(paste("OSPSuite CSV input does not exist:", standardize_csv_input))
} else {
  if (is.na(model_path)) stop("--model <pkml-json-or-pksim-file> is required")
  if (!file.exists(model_path)) stop(paste("OSP model file does not exist:", model_path))
}

convert_time_to_hours <- function(time_values, time_units) {
  normalized_units <- tolower(ifelse(is.na(time_units), "h", time_units))
  multiplier <- rep(NA_real_, length(normalized_units))
  multiplier[normalized_units %in% c("h", "hr", "hour", "hours")] <- 1.0
  multiplier[normalized_units %in% c("min", "minute", "minutes")] <- 1.0 / 60.0
  multiplier[normalized_units %in% c("s", "sec", "second", "seconds")] <- 1.0 / 3600.0
  if (any(is.na(multiplier))) {
    stop(paste("Unsupported OSPSuite time unit:", paste(unique(time_units[is.na(multiplier)]), collapse = ", ")))
  }
  as.numeric(time_values) * multiplier
}

standardize_simulation_results <- function(simulation_results) {
  result_frame <- as.data.table(simulationResultsToDataFrame(simulation_results))
  required_columns <- c("paths", "Time", "simulationValues")
  missing_columns <- setdiff(required_columns, names(result_frame))
  if (length(missing_columns) > 0) {
    stop(paste("OSPSuite result frame is missing columns:", paste(missing_columns, collapse = ", ")))
  }
  time_unit_column <- if ("TimeUnit" %in% names(result_frame)) result_frame$TimeUnit else rep("h", nrow(result_frame))
  data.table(
    source_output_path = result_frame$paths,
    time_hours = convert_time_to_hours(result_frame$Time, time_unit_column),
    concentration = as.numeric(result_frame$simulationValues)
  )
}

standardize_csv_file <- function(csv_path) {
  csv_frame <- fread(csv_path)
  if (all(c("paths", "Time", "simulationValues") %in% names(csv_frame))) {
    time_unit_column <- if ("TimeUnit" %in% names(csv_frame)) csv_frame$TimeUnit else rep("h", nrow(csv_frame))
    return(data.table(
      source_output_path = csv_frame$paths,
      time_hours = convert_time_to_hours(csv_frame$Time, time_unit_column),
      concentration = as.numeric(csv_frame$simulationValues)
    ))
  }
  time_column_candidates <- grep("^time|time", names(csv_frame), ignore.case = TRUE, value = TRUE)
  if (length(time_column_candidates) == 0) {
    stop(paste("OSPSuite CSV output lacks a time column:", csv_path))
  }
  time_column <- time_column_candidates[[1]]
  value_columns <- setdiff(names(csv_frame), time_column)
  if (length(value_columns) == 0) {
    stop(paste("OSPSuite CSV output lacks concentration columns:", csv_path))
  }
  time_unit <- ifelse(grepl("\\[.*\\]", time_column), sub(".*\\[(.*)\\].*", "\\1", time_column), "h")
  metadata_columns <- intersect(value_columns, csv_metadata_columns)
  candidate_measure_columns <- setdiff(value_columns, metadata_columns)
  measure_columns <- candidate_measure_columns[grepl(measure_column_keep_regex, candidate_measure_columns, ignore.case = TRUE)]
  if (length(measure_columns) == 0) {
    stop(paste("OSPSuite CSV output lacks concentration measure columns after metadata and keep-regex selection:", csv_path))
  }
  non_numeric_columns <- measure_columns[!vapply(measure_columns, function(column_name) {
    converted_values <- suppressWarnings(as.numeric(csv_frame[[column_name]]))
    all(is.na(csv_frame[[column_name]]) | !is.na(converted_values))
  }, logical(1))]
  if (length(non_numeric_columns) > 0) {
    stop(paste(
      "OSPSuite CSV output contains non-numeric measure columns not declared as metadata:",
      paste(non_numeric_columns, collapse = ", ")
    ))
  }
  measure_chunks <- split(
    measure_columns,
    ceiling(seq_along(measure_columns) / max_measure_columns_per_chunk)
  )
  melted <- rbindlist(lapply(measure_chunks, function(chunk_columns) {
    melt(
      csv_frame,
      id.vars = time_column,
      measure.vars = chunk_columns,
      variable.name = "source_output_path",
      value.name = "concentration"
    )
  }), fill = TRUE)
  data.table(
    source_output_path = paste(basename(dirname(csv_path)), melted$source_output_path, sep = "|"),
    time_hours = convert_time_to_hours(melted[[time_column]], rep(time_unit, nrow(melted))),
    concentration = as.numeric(melted$concentration)
  )
}

standardize_snapshot_outputs <- function(snapshot_path) {
  output_directory <- tempfile("osp_snapshot_outputs_")
  dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
  runSimulationsFromSnapshot(snapshot_path, output = output_directory, RunForAllOutputs = TRUE, exportCSV = TRUE, exportPKML = FALSE, exportJSON = FALSE, exportXML = FALSE)
  csv_files <- list.files(output_directory, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE)
  result_csv_files <- csv_files[grepl(result_file_regex, basename(csv_files))]
  if (length(result_csv_files) == 0) {
    stop(paste("OSPSuite snapshot run produced no result CSV files matching", result_file_regex))
  }
  rbindlist(lapply(result_csv_files, standardize_csv_file), fill = TRUE)
}

run_project_snapshot <- function(project_path) {
  snapshot_directory <- tempfile("osp_snapshot_")
  dir.create(snapshot_directory, recursive = TRUE, showWarnings = FALSE)
  convertSnapshot(project_path, format = "snapshot", output = snapshot_directory)
  snapshot_files <- list.files(snapshot_directory, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
  if (length(snapshot_files) == 0) {
    stop("OSPSuite project conversion did not produce snapshot JSON files")
  }
  rbindlist(lapply(snapshot_files, standardize_snapshot_outputs), fill = TRUE)
}

run_pkml_simulation <- function(pkml_path) {
  simulation <- loadSimulation(pkml_path)
  if (!is.na(dose_parameter_path) && !is.na(dose_value)) {
    dose_parameter <- getParameter(dose_parameter_path, simulation)
    if (is.na(dose_unit)) {
      setParameterValues(parameters = dose_parameter, values = dose_value)
    } else {
      setParameterValues(parameters = dose_parameter, values = dose_value, units = dose_unit)
    }
  }
  standardize_simulation_results(runSimulation(simulation))
}

if (!is.na(standardize_csv_input)) {
  rows <- standardize_csv_file(standardize_csv_input)
} else {
  file_extension <- tolower(tools::file_ext(model_path))
  if (file_extension == "pkml") {
    rows <- run_pkml_simulation(model_path)
  } else if (file_extension == "json") {
    rows <- standardize_snapshot_outputs(model_path)
  } else if (file_extension == "pksim5") {
    rows <- run_project_snapshot(model_path)
  } else {
    stop(paste("Unsupported OSPSuite model extension:", file_extension))
  }
}
if (nrow(rows) == 0) stop("OSPSuite simulation produced no output rows")
rows[, standardization_policy_uid := standardization_policy_uid]
rows[, plasma_concentration := fifelse(grepl(plasma_regex, source_output_path, ignore.case = TRUE), concentration, NA_real_)]
rows[, csf_concentration := fifelse(grepl(csf_regex, source_output_path, ignore.case = TRUE), concentration, NA_real_)]
rows[, brain_isf_concentration := fifelse(grepl(brain_isf_regex, source_output_path, ignore.case = TRUE), concentration, NA_real_)]
if (all(is.na(rows$plasma_concentration))) stop("OSPSuite output lacks a plasma-like concentration path")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
fwrite(rows, output_path, sep = "\t")
