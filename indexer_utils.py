INDEX_COM_FILE = 'ready.index'
MODEL_COM_FILE = 'ready.model'


def set_index_com_file_not_ready():
    with open(INDEX_COM_FILE, 'w') as com_file:
        com_file.write('0')


def set_index_com_file_ready():
    with open(INDEX_COM_FILE, 'w') as com_file:
        com_file.write('1')


def check_index_com_file_ready():
    if not os.path.exists(INDEX_COM_FILE):
        set_index_com_file_not_ready()

    with open(INDEX_COM_FILE, 'r') as com_file:
        return bool(com_file.readline())


def set_model_com_file_not_ready():
    with open(MODEL_COM_FILE, 'w') as com_file:
        com_file.write('0')


def set_model_com_file_ready():
    with open(MODEL_COM_FILE, 'w') as com_file:
        com_file.write('1')


def check_model_com_file_ready():
    if not os.path.exists(MODEL_COM_FILE):
        set_index_com_file_not_ready()

    with open(MODEL_COM_FILE, 'r') as com_file:
        return bool(com_file.readline())

