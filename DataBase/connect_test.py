import psycopg2
from psycopg2 import OperationalError

try:
    connection = psycopg2.connect(
        database="testDataBase",
        user="postgres",
        password="3076817064",  # 确保密码是字符串
        host="192.168.1.6",
        port=5432
    )
    print("成功连接到数据库！")
    cursor = connection.cursor()
    # create_table_sql = """
    #         CREATE TABLE IF NOT EXISTS test_table (
    #             id SERIAL PRIMARY KEY,
    #             test1 INTEGER NOT NULL,
    #             test2 BIGINT NOT NULL,
    #             test3 VARCHAR(256) NOT NULL,
    #             created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    #         );
    #         """
    # cursor.execute("SELECT version();")
    # cursor.execute(create_table_sql)
    cursor.execute("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'test_table';
            """)
    columns = cursor.fetchall()
    print("____________________")
    # connection.commit()
except OperationalError as e:
    print(f"❌ 连接失败，完整错误信息: {e}")  # 打印完整错误
finally:
    if 'connection' in locals() and connection is not None:
        cursor.close()
        connection.close()