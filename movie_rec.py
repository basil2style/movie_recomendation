
import surprise from Reader, Dataset

# Define the format
reader = Reader(line_format='user item rating timestamp', sep='\t')

# Load data from the 