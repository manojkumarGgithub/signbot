import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Sender and Receiver Details
sender_email = "manojkumer844@gmail.com"
receiver_email = "22341A4515@gmrit.edu.in"
app_password = "odal faip qjab azsk"  # Use App Password, NOT your actual Gmail password

# Email Content
subject = "Test Email from Python"
body = "Hello, this is a test email sent via Python using Gmail SMTP_SSL!"

# Create Message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

try:
    # Connect to Gmail SMTP Server using SSL
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)  # Use SSL and port 465
    server.login(sender_email, app_password)  # Login with app password
    server.sendmail(sender_email, receiver_email, msg.as_string())  # Send email
    server.quit()

    print("Email sent successfully!")
except Exception as e:
    print(f"Failed to send email: {e}")
