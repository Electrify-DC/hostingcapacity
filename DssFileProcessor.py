class DssFileProcessor:
    """
    The DSS File Processor class processes the dss files to ensure they can be processed be OpenDss. OpenDss performs
    only steady state analysis. For example, it does not perform a harmonic study. Files from other platforms often
    include 4x4 matrices for 3 phase 4 wire systems however OpenDss does not use the 4th wire (neutral) as it is used
    for harmonic studies.
    """
    def __init__(self, master_file: str):
        self.master = master_file

    def get_file_content(self, filename: str) -> list[str]:
        """
        Gets all the lines from a file as a list of strings.
        :param filename: name of the file may include path
        :return: a list of strings with each item being a line in the file
        """
        with open(filename, 'r') as file:
            content = file.read()
            lines = content.splitlines()
        return lines

    def reduce_matrix(self, matrix: list[str], phases: int, splitter: str, end_str: str) -> str:
        """
        Performs matrix reduction on a symmetric matrix
        :param matrix: List of strings in matrix form to be reduced
        :param phases: number of rows and columns
        :param splitter: delimiter to split each line
        :param end_str: append at the end of the line
        :return: the matrix as a one line string
        """
        matrix = matrix[0:phases]
        new_row = ""
        for i in range(phases):
            row = matrix[i].split()[0:phases]
            rs = ' '.join(str(row) for row in row)
            if i == phases - 1:
                reduced_row = rs + end_str
            else:
                reduced_row = rs + splitter
            new_row = new_row + reduced_row
        return new_row

    def perform_kron_reduction(self, line_codes: list[str]):
        """
        Performs Kron Reduction on matrices representing network.txt resistance, reactance, impedance, etc.
        :param line_codes: Line codes from a dss file
        :return: Lines that have undergone kron reduction
        """
        new_line_codes = []
        for idx, line in enumerate(line_codes):
            if line == "":
                new_line = ""
            else:
                code = line.split("=")
                phases = int(code[2].split()[0])
                if len(code[3].split("|")) > phases:
                    # Reduce r and x matrices
                    rmatrix = code[3].split("|")
                    end_r_row = code[3].split(')')[-1]
                    # If there is no next parameter
                    if end_r_row == code[3].split(')')[0]:
                        end_r_row = ""
                    new_r_row = self.reduce_matrix(matrix=rmatrix, phases=phases, splitter=" | ",
                                                   end_str=f" ) {end_r_row}")
                    xmatrix = code[4].split("|")
                    end_x_row = code[4].split(')')[-1]
                    # If there is no next parameter
                    if end_x_row == code[4].split(')')[0]:
                        end_x_row = ""
                    new_x_row = self.reduce_matrix(matrix=xmatrix, phases=phases, splitter=" | ",
                                                   end_str=f" ) {end_x_row}")
                    # Replace with reduced matrices
                    code[3] = new_r_row
                    code[4] = new_x_row
                    new_line = "=".join(code)
                else:
                    new_line = line
            new_line_codes.append(new_line + "\n")
        return new_line_codes

    def process_master_files(self):
        """
        Process the Master Dss File and the encoded files within it.
        :return: Clean Master Dss file readable by python's opendss
        """
        master_lines = self.get_file_content(self.master)
        for line in master_lines:
            if "LineCodes.dss" in line:
                # filename = os.path.join(self.data_folder_path, self.network_filename)
                line_code_file = line.split()[1].strip()
                line_codes = self.get_file_content(line_code_file)
                break
        new_line_codes = self.perform_kron_reduction(line_codes)
        with open(line_code_file, 'w') as f:
            f.writelines(new_line_codes)
