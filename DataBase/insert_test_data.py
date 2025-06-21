import psycopg2

from DataBase.gene_test_data import GeneData
from DataBase.setting import Config


class InsertData:
    def __init__(self):
        self.config = Config()
        self.connection = psycopg2.connect(
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port
        )
        self.geneData = GeneData()

    def insert_employee_data(self):
        cursor = self.connection.cursor()
        cursor.executemany("INSERT INTO employee (name, age, gender, departmentid, salary) VALUES (%s, %s, %s, %s, %s)",
                           self.geneData.get_employee_data())
        self.connection.commit()
        cursor.close()

    def insert_data(self):
        self.insert_employee_data()

    def close_connection(self):
        self.connection.close()


if __name__ == '__main__':
    insertData = InsertData()
    insertData.insert_data()
    insertData.close_connection()
    print("_________________________")
