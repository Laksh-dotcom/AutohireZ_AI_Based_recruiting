import os
from smtplib import SMTP
from dotenv import load_dotenv

load_dotenv()
class Details:
    def __init__(self, name, gmail, message):
        self.user = name
        self.gmail = gmail
        self.message = message
        self.mail = os.environ.get("MAIL")
        self.password = os.environ.get("PASSWORD")

    def send(self):
        with SMTP("smtp.gmail.com", 587) as connection:
            connection.starttls()
            connection.login(user=self.mail, password=self.password)
            connection.sendmail(
                from_addr=self.mail,
                to_addrs=self.gmail,
                msg=f"Subject:Hello {self.user}\n\n{self.message}"
            )

