# Cybershsake_Partition

## Program structure

```
├── inputData/           # Contains the input parameter file (e.g., parameters.in)
├── Partition_utils/     # Utility functions for handling velocity models
├── run_Partition.sh     # Shell script to run the partitioning process using inputData/parameters.in
```

## Running the Example

You can run the partitioning process using the provided shell script:

```bash
./run_Partition.sh  # Executes the partitioning with parameters defined in inputData/parameters.in
```

### Output

Upon execution, the script generates the **outputData/Partition** directory, which contains:

- `parameters_partition.in`, A copy of the input configuration file used during execution.
- `metadata_partition.json`, Metadata describing the partitioning process.
- `partition_otheroptions.out`, Alternative core decomposition triplets.
- `partition_results.out`, Results of the partitioning process, highlighting the optimal partition found.


## Notes

- Ensure the required input file (`parameters.in`) is present in the `inputData/` directory before running the script.
- Make sure all variables in `parameters.in` are properly filled. 
- It is recommended to delete the existing `outputData/` directory before re-running.

## License
This project is released under the BSD 3-Clause open-source license.

## Acknowledgment
This is an open-source project developed by Andrea C. Riaño (andrea.riano@bsc.es).
Version 1.0.0, released in October 2026.
