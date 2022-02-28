# 準備訊息物件設定
# 載入模組
import smtplib
import email.message
# 建立訊息物件
msg = email.message.EmailMessage()
# 利用物件建立基本設定

from_a = input("請輸入寄件人信箱：")
to_b = input("請輸入收件人信箱：")


msg["From"] = from_a
msg["To"] = to_b
msg["Subject"] = "你好"

# 寄送郵件主要內容
# msg.set_content("測試郵件純文字內容") #純文字信件內容
msg.add_alternative("<h3>HTML內容</h3>安安這是寄送郵件測試", subtype="html")  # HTML信件內容

acc = input("請輸入gmail帳號：")
password = input("請輸入密碼")

# 連線到SMTP Sevver
# 可以從網路上找到主機名稱和連線埠
server = smtplib.SMTP_SSL("smtp.gmail.com", 465)  # 建立gmail連驗
server.login(acc, password)
server.send_message(msg)
server.close()  # 發送完成後關閉連線
