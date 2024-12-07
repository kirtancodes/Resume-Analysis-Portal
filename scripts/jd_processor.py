def read_jd_file(jd_path):
    """
    Read the job description file and extract its text.
    """
    with open(jd_path, 'r') as file:
        return file.read()
