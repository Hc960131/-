import random


def get_random_age():
    return random.randint(25, 51)


def get_random_gender():
    return random.choice(['man', 'woman'])


def get_random_salary():
    return random.randint(5000, 50001)


def get_language():
    return random.choice(['java', 'python', 'c++'])


def get_sale():
    return random.randint(100000, 100000000)


class GeneData:
    def __init__(self):
        self.name = 'employee'
        self.sale_part = 200
        self.manage_part = 100
        self.length = 1000
        self.project_num = 100
        self.client = 10

    def get_employee_data(self):
        result = []
        for i in range(self.sale_part):
            result.append([self.name + str(i), get_random_age(), get_random_gender(), 1, get_random_salary()])
        for i in range(self.sale_part, self.sale_part + self.manage_part):
            result.append([self.name + str(i), get_random_age(), get_random_gender(), 2, get_random_salary()])
        for i in range(self.sale_part + self.manage_part, self.length):
            result.append([self.name + str(i), get_random_age(), get_random_gender(), 3, get_random_salary()])
        return result

    def get_developer_data(self):
        result = []
        for i in range(self.sale_part + self.manage_part + 1, self.length + 1):
            result.append([i, get_language()])
        return result

    def get_project_data(self):
        result = []
        sale = get_sale()
        for i in range(self.project_num):
            result.append([get_language(), self.get_random_sale(), self.get_random_project_manager(),
                           self.get_random_project_client(), sale, int(sale * 0.8)])

    def get_random_sale(self):
        return random.choice(range(self.sale_part))

    def get_random_project_manager(self):
        return random.choice(range(self.sale_part + self.manage_part, self.length))

    def get_random_project_client(self):
        return random.choice(range(self.client))


if __name__ == '__main__':
    pass
