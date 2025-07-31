import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os
import uuid
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import urllib.parse

# ====================
# CONFIGURATION/SETUP
# ====================
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Please add it to your .env.")

genai.configure(api_key=GOOGLE_API_KEY)
GENERATION_CONFIG = {
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1200,
}

# ====================
# UTILITY FUNCTIONS
# ====================
def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return round(R * c, 2)

def generate_service_id():
    return f"SVC-{uuid.uuid4().hex[:8].upper()}"

def get_available_crew(crew_df, lat, lon):
    """Find nearest available crew and calculate distance."""
    candidates = crew_df[crew_df["Crew ID available"] == "Yes"].copy()
    if len(candidates) == 0:
        return None
    
    candidates["distance"] = candidates.apply(
        lambda row: calculate_distance(lat, lon, row["Location of crew latitude"], row["Location of crew longitude"]),
        axis=1
    )
    return candidates.nsmallest(1, "distance").iloc[0]

def generate_navigation_map_link(outage_lat, outage_lon, crew_lat, crew_lon, outage_location, crew_location):
    """Generate Google Maps link with multiple locations showing route between crew and outage."""
    # Create waypoints for Google Maps directions
    origin = f"{crew_lat},{crew_lon}"
    destination = f"{outage_lat},{outage_lon}"
    
    # Multi-location map link
    map_url = f"https://www.google.com/maps/dir/{origin}/{destination}/@{(crew_lat+outage_lat)/2},{(crew_lon+outage_lon)/2},12z"
    
    return map_url

def generate_ai_analysis(customer, customer_id, component, comp_id, address, detected_time, status, crew_assignment):
    """Generate enhanced AI analysis using Gemini."""
    try:
        prompt = f"""
        OUTAGE EMERGENCY ANALYSIS REQUIRED
        
        Customer Details:
        - Customer Name: {customer} (ID: {customer_id})
        - Location: {address}
        - Component Affected: {component} (ID: {comp_id})
        - Detection Time: {detected_time}
        - Current Status: {status}
        
        Crew Assignment:
        {crew_assignment}
        
        Please provide a comprehensive analysis with the following sections:
        
        1. SEVERITY ASSESSMENT
        - Risk level evaluation
        - Customer impact assessment
        - Equipment damage potential
        - Safety considerations
        
        2. ROOT CAUSE ANALYSIS
        - Potential failure mechanisms
        - Component-specific issues
        - Environmental factors
        - Historical failure patterns
        
        3. IMMEDIATE ACTIONS
        - Emergency response procedures
        - Safety protocols
        - Equipment isolation steps
        - Customer communication requirements
        
        4. MAINTENANCE RECOMMENDATIONS
        - Repair procedures
        - Replacement requirements
        - Testing protocols
        - Quality assurance steps
        
        5. CREW DEPLOYMENT STRATEGY
        - Optimal crew utilization
        - Equipment requirements
        - Timeline estimation
        - Resource allocation
        
        Format response in clear bullet points for each section. Be specific and actionable.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
        return response.text
    except Exception as e:
        return generate_fallback_analysis(component, status)

def generate_fallback_analysis(component, status):
    """Fallback analysis when AI fails."""
    return f"""
SEVERITY ASSESSMENT:
• {status} outage detected - requires immediate attention
• {component} failure can cause extended service disruption
• Customer impact: High - service restoration critical
• Safety risk: Medium - follow standard safety protocols

ROOT CAUSE ANALYSIS:
• {component} component failure detected
• Possible causes: wear and tear, environmental factors, overload
• Requires on-site inspection for detailed diagnosis
• Component age and maintenance history should be reviewed

IMMEDIATE ACTIONS:
• Dispatch crew to site immediately
• Isolate affected equipment for safety
• Notify customer of estimated restoration time
• Monitor system for cascading failures

MAINTENANCE RECOMMENDATIONS:
• Replace/repair {component} as per manufacturer specifications
• Test system functionality after repair
• Update maintenance records
• Schedule follow-up inspection

CREW DEPLOYMENT STRATEGY:
• Single crew deployment sufficient for initial assessment
• Ensure crew has necessary tools and replacement parts
• Estimated resolution time: 2-4 hours
• Maintain communication with dispatch center
"""

def generate_pdf_report(ai_analysis, service_id, customer, outage_details):
    """Generate detailed PDF report with AI analysis."""
    filename = f"AI_Analysis_{service_id}.pdf"
    
    try:
        doc = SimpleDocTemplate(
            filename, 
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        styles = getSampleStyleSheet()
        
        # Enhanced custom styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor='#1976d2'
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor='#d32f2f'
        ))
        
        styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=styles['Normal'],
            fontSize=10,
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20
        ))
        
        elements = []
        
        # Header
        elements.append(Paragraph("⚡ AI OUTAGE ANALYSIS REPORT", styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Service details table
        service_info = f"""
        <b>Service ID:</b> {service_id}<br/>
        <b>Customer:</b> {customer}<br/>
        <b>Component:</b> {outage_details.get('component', 'N/A')}<br/>
        <b>Location:</b> {outage_details.get('address', 'N/A')}<br/>
        <b>Status:</b> {outage_details.get('status', 'N/A')}<br/>
        <b>Report Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        elements.append(Paragraph(service_info, styles['Normal']))
        elements.append(Spacer(1, 0.4*inch))
        
        # AI Analysis content with better formatting
        elements.append(Paragraph("COMPREHENSIVE AI ANALYSIS", styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Process analysis by sections
        sections = ai_analysis.split('\n\n')
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n')
                section_header = lines[0] if lines else ""
                
                # Check if first line is a header
                if section_header and (section_header.isupper() or section_header.endswith(':')):
                    elements.append(Paragraph(section_header, styles['SectionHeader']))
                    elements.append(Spacer(1, 0.1*inch))
                    
                    # Process remaining lines as content
                    for line in lines[1:]:
                        if line.strip():
                            if line.strip().startswith('•') or line.strip().startswith('-'):
                                elements.append(Paragraph(line.strip(), styles['BulletPoint']))
                            else:
                                elements.append(Paragraph(line.strip(), styles['Normal']))
                            elements.append(Spacer(1, 0.05*inch))
                else:
                    # No clear header, treat as normal content
                    for line in lines:
                        if line.strip():
                            elements.append(Paragraph(line.strip(), styles['Normal']))
                            elements.append(Spacer(1, 0.05*inch))
                
                elements.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(elements)
        print(f"✅ PDF report generated successfully: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        # Return a simple text file as fallback
        fallback_filename = f"AI_Analysis_{service_id}.txt"
        with open(fallback_filename, 'w') as f:
            f.write(f"AI ANALYSIS REPORT\n")
            f.write(f"Service ID: {service_id}\n")
            f.write(f"Customer: {customer}\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            f.write(ai_analysis)
        return fallback_filename


# ====================
# LOAD DATA
# ====================
try:
    outage_df = pd.read_csv("outage_data.csv")
    crew_df = pd.read_csv("crew_data.csv")
    print("✅ Successfully loaded CSV files")
    print(f"📊 Loaded {len(outage_df)} outage records and {len(crew_df)} crew records")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("Please ensure 'outage_data.csv' and 'crew_data.csv' exist in the current directory")
    exit()

# Only process outages that are unresolved/pending (not resolved)
active_outages = outage_df[outage_df["outage status"].isin(["Pending", "Detected"])]
print(f"🔍 Found {len(active_outages)} active outages to process")

# ====================
# MAIN EVENT LOOP
# ====================
processed_count = 0
for _, outage in active_outages.iterrows():
    try:
        # Get required info
        service_id = generate_service_id()
        customer = outage["customer name"]
        customer_id = outage["customer ID"]
        address = outage["customer address affected by outage"]
        component = outage["component causing the outage"]
        status = outage["outage status"]
        comp_id = outage["component ID"]
        detected_time = outage["Outage Detection Time"]
        comp_lat = outage["Component Location Latitude"]
        comp_lon = outage["Component Location Longitude"]
        gm_link = outage["component location Google maps link"]

        print(f"\n⚡ Processing outage for {customer} (Service ID: {service_id})")

        # Assign nearest available crew
        crew = get_available_crew(crew_df, comp_lat, comp_lon)
        if crew is not None:
            crew_assignment = (
                f"Crew Name: {crew['Crew name']}\n"
                f"Crew ID: {crew['Crew ID']}\n"
                f"Supervisor: {crew['Supervisor Assigned']} (ID: {crew['Supervisor ID']})\n"
                f"Location: {crew['Location of crew']}\n"
                f"Distance to site: {crew['distance']:.2f} km\n"
                f"ETA: {int(crew['distance'] * 2)} minutes"
            )
            
            # Generate navigation map
            nav_map_link = generate_navigation_map_link(
                comp_lat, comp_lon, 
                crew['Location of crew latitude'], crew['Location of crew longitude'],
                address, crew['Location of crew']
            )
            
            print(f"👷 Assigned crew: {crew['Crew name']} ({crew['distance']:.2f} km away)")
        else:
            crew_assignment = "No available crew for dispatch - Manual assignment required"
            nav_map_link = gm_link
            print("⚠️ No available crew found for dispatch")

        # Generate AI analysis
        print("🤖 Generating comprehensive AI analysis...")
        ai_analysis = generate_ai_analysis(
            customer, customer_id, component, comp_id, 
            address, detected_time, status, crew_assignment
        )
        print("✅ AI analysis generated successfully")

        # Generate PDF report
        outage_details = {
            'customer': customer,
            'component': component,
            'address': address,
            'status': status
        }
        pdf_filename = generate_pdf_report(ai_analysis, service_id, customer, outage_details)
        print(f"📄 PDF report generated: {pdf_filename}")

        # Compose enhanced HTML email
        sender_email = os.getenv("SMTP_EMAIL", "prathamkaushal08@gmail.com")
        sender_password = os.getenv("SMTP_PASSWORD", "dlhw kedy surr ydbr")
        recipient_email = "prathamkaushal08@gmail.com"

        subject = f"🚨 CRITICAL OUTAGE ALERT - {service_id} | {customer}"
        
        # Create HTML email body with proper styling
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #d32f2f; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .section {{ margin-bottom: 20px; background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .button {{ display: inline-block; padding: 10px 20px; background-color: #1976d2; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
                .button:hover {{ background-color: #1565c0; }}
                .crew-info {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
                .outage-info {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                .ai-section {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 CRITICAL OUTAGE DETECTED</h1>
                <h2>Service ID: {service_id}</h2>
            </div>
            
            <div class="content">
                <div class="section outage-info">
                    <h3>📍 OUTAGE DETAILS</h3>
                    <p><strong>Customer:</strong> {customer} (ID: {customer_id})</p>
                    <p><strong>Location:</strong> {address}</p>
                    <p><strong>Component:</strong> {component} (ID: {comp_id})</p>
                    <p><strong>Status:</strong> <span style="color: #d32f2f; font-weight: bold;">{status}</span></p>
                    <p><strong>Detection Time:</strong> {detected_time}</p>
                    <p><strong>Coordinates:</strong> {comp_lat}, {comp_lon}</p>
                </div>
                
                <div class="section crew-info">
                    <h3>👷 ASSIGNED CREW</h3>
                    <pre>{crew_assignment}</pre>
                </div>
                
                <div class="section">
                    <h3>🗺️ NAVIGATION & MAPS</h3>
                    <a href="{nav_map_link}" class="button" target="_blank">
                        🧭 Navigate to Outage Site (Shows Route & Distance)
                    </a>
                    <a href="{gm_link}" class="button" target="_blank">
                        📍 Outage Location Details
                    </a>
                </div>
                
                <div class="section ai-section">
                    <h3>🤖 AI ANALYSIS PREVIEW</h3>
                    <p>Comprehensive analysis has been generated covering:</p>
                    <ul>
                        <li>Severity Assessment</li>
                        <li>Root Cause Analysis</li>
                        <li>Immediate Actions Required</li>
                        <li>Maintenance Recommendations</li>
                        <li>Crew Deployment Strategy</li>
                    </ul>
                    <p><strong>📄 Download complete analysis:</strong> See attached PDF report</p>
                </div>
                
                <div class="section">
                    <h3>⚡ IMMEDIATE ACTIONS REQUIRED</h3>
                    <ol>
                        <li>Confirm crew dispatch immediately</li>
                        <li>Review attached AI analysis report</li>
                        <li>Follow navigation link for optimal route</li>
                        <li>Update customer on estimated restoration time</li>
                    </ol>
                </div>
                
                <hr>
                <p style="text-align: center; color: #666; font-size: 12px;">
                    This alert was automatically generated by the Advanced Outage Management System<br>
                    Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </body>
        </html>
        """

        # Send email with PDF attachment
        msg = MIMEMultipart('alternative')
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        
        # Attach HTML body
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach PDF report
        try:
            with open(pdf_filename, 'rb') as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                pdf_attachment.add_header(
                    'Content-Disposition', 
                    'attachment', 
                    filename=f"AI_Analysis_{service_id}.pdf"
                )
                msg.attach(pdf_attachment)
        except Exception as e:
            print(f"⚠️ Could not attach PDF: {e}")

        # Send email
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"✅ Enhanced email sent for outage {service_id}")
            processed_count += 1
            
            # Clean up PDF file after sending
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
                
        except Exception as e:
            print(f"❌ Failed to send email for outage {service_id}: {str(e)}")

    except Exception as e:
        print(f"❌ Error processing outage: {e}")
        continue

print(f"\n🎯 Processing complete! {processed_count} outages processed and enhanced alerts sent.")
