import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import uuid
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini AI and model
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file")

# Initialize Gemini with safety settings
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the model with updated parameters
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

class FaultDetectionSystem:
    def __init__(self, email_recipient="prathamkaushal08@gmail.com"):
        self.email_recipient = email_recipient
        # Reduce contamination to 0.02 (2%) to detect fewer anomalies
        self.anomaly_detector = IsolationForest(contamination=0.02, random_state=42)
        
    def generate_sample_data(self, n_samples=100):  # Reduced from 1000 to 100 samples
        """Generate pseudo sensor data for testing"""
        np.random.seed(42)
        
        # Generate normal sensor readings
        data = {
            'service_id': range(1, n_samples + 1),
            'temperature': np.random.normal(25, 2, n_samples),  # Reduced variance
            'pressure': np.random.normal(100, 3, n_samples),    # Reduced variance
            'vibration': np.random.normal(50, 1, n_samples),    # Reduced variance
            'timestamp': [datetime.datetime.now() - datetime.timedelta(minutes=x) for x in range(n_samples)]
        }
        

        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
        for idx in anomaly_indices:
            data['temperature'][idx] *= 2.0  # More extreme anomalies
            data['pressure'][idx] *= 1.8     # More extreme anomalies
            data['vibration'][idx] *= 1.9    # More extreme anomalies
            
        return pd.DataFrame(data)

    def detect_faults(self, data):
        """Detect faults in sensor data using Isolation Forest"""
        features = ['temperature', 'pressure', 'vibration']
        X = data[features]
        
        # Fit and predict anomalies
        predictions = self.anomaly_detector.fit_predict(X)
        
        # Mark anomalies (-1 indicates anomaly, 1 indicates normal)
        anomalies = data[predictions == -1]
        return anomalies

    def generate_fault_service_id(self):
        """Generate unique alphanumeric service ID for fault detection"""
        return f"FLT-{uuid.uuid4().hex[:8].upper()}"

    def generate_ai_analysis(self, fault_data):
        """Generate AI analysis of the faults using Gemini"""
        try:
            # Create model instance using the correct model name
            model = genai.GenerativeModel('gemini-1.5-flash')  # Changed from gemini-1.0-pro
            
            # Prepare fault data for AI analysis
            fault_description = f"Analyzing {len(fault_data)} sensor faults:\n"
            for _, row in fault_data.iterrows():
                fault_description += f"Temperature: {row['temperature']:.2f}°C, "
                fault_description += f"Pressure: {row['pressure']:.2f}psi, "
                fault_description += f"Vibration: {row['vibration']:.2f}Hz\n"

            prompt = f"""
            As an industrial sensor fault analysis expert, analyze these sensor readings and provide:
            1. Potential causes of the anomalies
            2. Severity assessment
            3. Recommended actions
            
            Sensor Data:
            {fault_description}
            """
            
            # Generate response with safety settings
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            )
            
            # Check and return response
            if response and hasattr(response, 'text'):
                return response.text
            return "No analysis generated"
            
        except Exception as e:
            print(f"Error generating AI analysis: {str(e)}")
            return f"AI analysis unavailable at the moment. Error: {str(e)}"

    def send_email_alert(self, fault_data):
        """Send email alert for detected faults with AI analysis"""
        sender_email = "prathamkaushal08@gmail.com"
        sender_password = "dlhw kedy surr ydbr"
        
        # Generate unique service ID for this fault detection event
        fault_service_id = self.generate_fault_service_id()
        
        # Get AI analysis
        ai_analysis = self.generate_ai_analysis(fault_data)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = self.email_recipient
        msg['Subject'] = f"Fault Detection Alert - Service ID: {fault_service_id}"
        
        # Enhanced message body with AI analysis
        body = f"Fault Detection Service ID: {fault_service_id}\n"
        body += f"Detection Timestamp: {datetime.datetime.now()}\n"
        body += f"Number of Faults Detected: {len(fault_data)}\n\n"
        body += "AI Analysis:\n" + "="*50 + "\n"
        body += ai_analysis + "\n\n"
        body += "Detailed Fault Information:\n" + "="*50 + "\n\n"
        
        for _, row in fault_data.iterrows():
            body += f"Sensor ID: {row['service_id']}\n"
            body += f"Timestamp: {row['timestamp']}\n"
            body += f"Temperature: {row['temperature']:.2f}\n"
            body += f"Pressure: {row['pressure']:.2f}\n"
            body += f"Vibration: {row['vibration']:.2f}\n"
            body += "-" * 50 + "\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print(f"Alert email sent successfully! Service ID: {fault_service_id}")
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            
        finally:
            server.quit()
        
        return fault_service_id

def main():
    # Initialize the fault detection system
    fault_system = FaultDetectionSystem()
    
    # Generate sample sensor data
    sensor_data = fault_system.generate_sample_data()
    print("Generated sample sensor data...")
    
    # Detect faults
    detected_faults = fault_system.detect_faults(sensor_data)
    print(f"Detected {len(detected_faults)} potential faults")
    
    # If faults are detected, send email alert
    if not detected_faults.empty:
        service_id = fault_system.send_email_alert(detected_faults)
        print(f"Fault report generated with Service ID: {service_id}")
    else:
        print("No faults detected")

if __name__ == "__main__":
    main()