# ---- Preprocess Function ----
def pre_fun(patch, **kargs):
    """
    Convert DAS optical phase data to strain rate using physical parameters.

    Args:
        patch: DAS patch object containing optical phase data (e.g., from DAS measurements [time, space]).
        **kargs: Additional optional keyword arguments (currently unused).

    Returns:
        Updated patch with strain-rate data in strain units.
    """
    # Physical constants and parameters for conversion
    ri = 1.4682           # Refractive index of the fiber
    gl = 16               # Gauge length in meters
    psi = 0.79            # Polarization factor (depends on fiber and interrogator)
    lmbda = 1550.12e-9    # Laser wavelength in meters (e.g., 1550 nm)

    # Calculate conversion factor from phase change to strain
    pha2str_factor = lmbda / (4 * np.pi * ri * gl * psi)

    # Extract raw data (phase measurements)
    data = patch.data

    # Optional: Remove common mode noise (disabled but can be used if needed)
    # common_mode = np.mean(data, axis=0, keepdims=True)

    # Compute strain rate as the time derivative (gradient along time axis)
    strain_rate = np.gradient(data, axis=0)

    # Create a new patch with strain-rate data
    patch = patch.new(data=strain_rate)

    # Apply phase-to-strain conversion factor
    return patch * pha2str_factor
# ---- One Worker for a Single Time Chunk ----
def process_time_chunk(args):
    """
    Process a single time chunk of DAS data:
    - Loads the data from the specified directory.
    - Selects the portion of data within [t_start, t_end].
    - Applies preprocessing (phase-to-strain rate conversion).
    - Computes standard deviation over time windows of specified duration (patch_size).
    - Saves the results to an output subfolder.

    Args:
        args (tuple): (datadir, t_start, t_end, output_path)
            datadir (str): Path to DAS dataset (directory of spools).
            t_start (np.datetime64): Start time for this chunk.
            t_end (np.datetime64): End time for this chunk.
            output_path (str): Directory to save the processed chunk output.

    Behavior:
        - Creates a subfolder for each chunk named as <start>_to_<end>.
        - Skips processing if no data is found in the given time range.
        - Logs errors instead of crashing on failure.

    Notes:
        - `patch_size` in `proc.std` is given in seconds, and defines the time window
          used to compute the standard deviation within the selected chunk.
    """
    datadir, t_start, t_end, output_path = args
    try:
        # Open DAS data spool (directory of DAS files)
        sp = dc.spool(datadir)

        # Select only the subset of data within the given time range
        sp = sp.select(time=(t_start, t_end))

        # If no data exists in this time range, skip processing
        if len(sp) == 0:
            return

        # Prepare subfolder for saving this chunk's output
        subfolder = os.path.join(output_path, f"{str(t_start).replace(':','-')}_to_{str(t_end).replace(':','-')}")
        os.makedirs(subfolder, exist_ok=True)

        # Compute standard deviation on the DAS patches
        # patch_size=1 → 1 second window for std calculation
        # pre_process=pre_fun → apply phase-to-strain conversion
        # overwrite=True → replace if output already exists
        proc.std(sp, subfolder, patch_size=1, pre_process=pre_fun, overwrite=True)

    except Exception as e:
        # Catch any error for this chunk and report, but don't interrupt other chunks
        print(f"[Error] Chunk {t_start}–{t_end} failed: {e}")
# ---- Main Parallel Wrapper ----
def run_parallel_std(datadir, output_folder, start_time_str, end_time_str, chunk_minutes=30):
    """
    Splits a large DAS dataset into time chunks and processes them in parallel.

    Workflow:
        1. Converts the start and end time strings into np.datetime64 objects.
        2. Creates a list of time chunks of length `chunk_minutes`.
        3. Launches multiple worker processes to handle each time chunk in parallel.
        4. Displays a progress bar for monitoring.

    Args:
        datadir (str): Path to the directory containing DAS data (Spool).
        output_folder (str): Directory where processed results will be saved.
        start_time_str (str): Start time in 'YYYY-MM-DD HH:MM:SS' format.
        end_time_str (str): End time in 'YYYY-MM-DD HH:MM:SS' format.
        chunk_minutes (int): Length of each processing chunk in minutes (default = 30).

    Behavior:
        - For each chunk:
            * Data is read from `datadir` for the specified time range.
            * Processed via `process_time_chunk` (which applies phase-to-strain conversion
              and computes standard deviation using `proc.std`).
            * Saves results in a subfolder for that time chunk.
        - Utilizes all available CPU cores via multiprocessing.Pool.
        - Skips empty time ranges gracefully without errors.

    Notes:
        - Multiprocessing uses `imap_unordered` for efficiency.
        - tqdm is used to display a single progress bar for all chunks.
    """
    # Convert start and end times to numpy datetime objects
    start_time = np.datetime64(start_time_str)
    end_time = np.datetime64(end_time_str)

    # Generate a list of time chunk tuples (datadir, start_time, end_time, output_folder)
    t = start_time
    time_chunks = []
    while t < end_time:
        # Compute end of current chunk
        t_next = t + np.timedelta64(chunk_minutes, 'm')
        # Ensure the last chunk does not exceed the end time
        time_chunks.append((datadir, t, min(t_next, end_time), output_folder))
        t = t_next

    print(f"Launching {len(time_chunks)} chunks with {mp.cpu_count()} workers...")

    # Create a multiprocessing pool using all CPU cores
    # Use tqdm to track overall progress for all chunks
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_time_chunk, time_chunks),
                      total=len(time_chunks), desc="Processing DAS chunks"):
            pass
