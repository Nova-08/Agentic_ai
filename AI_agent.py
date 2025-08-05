
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

# ========== CONFIGURATION ==========
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

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return round(R * c, 2)

def generate_service_id():
    return f"SVC-{uuid.uuid4().hex[:8].upper()}"

def get_available_crew(crew_df, lat, lon):
    candidates = crew_df[crew_df["Crew ID available"] == "Yes"].copy()
    if len(candidates) == 0:
        return None
    candidates["distance"] = candidates.apply(
        lambda row: calculate_distance(lat, lon, row["Location of crew latitude"], row["Location of crew longitude"]),
        axis=1
    )
    return candidates.nsmallest(1, "distance").iloc[0]

def osm_static_map_url(comp_lat, comp_lon, crew_lat, crew_lon, width=650, height=350):
    base_url = "https://staticmap.openstreetmap.de/staticmap.php"
    marker_crew = f"markers={crew_lat},{crew_lon},lightblue1"
    marker_outage = f"markers={comp_lat},{comp_lon},red-pushpin"
    center_lat = (float(comp_lat) + float(crew_lat)) / 2
    center_lon = (float(comp_lon) + float(crew_lon)) / 2
    params = (
        f"center={center_lat},{center_lon}"
        f"&zoom=11&size={width}x{height}"
        f"&{marker_crew}&{marker_outage}"
    )
    return f"{base_url}?{params}"

def google_maps_directions_link(comp_lat, comp_lon, crew_lat, crew_lon):
    return f"https://www.google.com/maps/dir/{crew_lat},{crew_lon}/{comp_lat},{comp_lon}/"

def generate_ai_analysis(customer, customer_id, component, comp_id, address, detected_time, status, crew_assignment):
    try:
        prompt = f"""OUTAGE EMERGENCY ANALYSIS REQUIRED
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
2. ROOT CAUSE ANALYSIS
3. IMMEDIATE ACTIONS
4. MAINTENANCE RECOMMENDATIONS
5. CREW DEPLOYMENT STRATEGY
Format response in clear bullet points for each section. Be specific and actionable.
"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
        return response.text
    except Exception:
        return generate_fallback_analysis(component, status)

def generate_fallback_analysis(component, status):
    return f"""
SEVERITY ASSESSMENT:
• {status} outage detected - requires immediate attention
• {component} failure can cause service disruption

ROOT CAUSE ANALYSIS:
• Possible wear/tear or overload; site inspection advised

IMMEDIATE ACTIONS:
• Dispatch crew; isolate equipment; notify customer

MAINTENANCE RECOMMENDATIONS:
• Replace/repair {component}; test/monitor

CREW DEPLOYMENT STRATEGY:
• Assign nearest crew with adequate tools
"""

def generate_pdf_report(ai_analysis, service_id, customer, outage_details):
    filename = f"AI_Analysis_{service_id}.pdf"
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle', parent=styles['Heading1'],
            fontSize=18, spaceAfter=30, alignment=1, textColor='#1976d2'
        ))
        styles.add(ParagraphStyle(
            name='SectionHeader', parent=styles['Heading2'],
            fontSize=14, spaceBefore=20, spaceAfter=10, textColor='#d32f2f'
        ))
        styles.add(ParagraphStyle(
            name='BulletPoint', parent=styles['Normal'],
            fontSize=10, spaceBefore=5, spaceAfter=5, leftIndent=20
        ))
        elements = []
        elements.append(Paragraph("⚡ AI OUTAGE ANALYSIS REPORT", styles['CustomTitle']))
        elements.append(Spacer(1, 0.3 * inch))
        service_info = f"""
        <b>Service ID:</b> {service_id}<br/>
        <b>Customer:</b> {customer}<br/>
        <b>Component:</b> {outage_details.get('component', 'N/A')}<br/>
        <b>Location:</b> {outage_details.get('address', 'N/A')}<br/>
        <b>Status:</b> {outage_details.get('status', 'N/A')}<br/>
        <b>Report Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        elements.append(Paragraph(service_info, styles['Normal']))
        elements.append(Spacer(1, 0.4 * inch))
        elements.append(Paragraph("COMPREHENSIVE AI ANALYSIS", styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))
        sections = ai_analysis.split('\n\n')
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n')
                section_header = lines[0] if lines else ""
                if section_header and (section_header.isupper() or section_header.endswith(':')):
                    elements.append(Paragraph(section_header, styles['SectionHeader']))
                    elements.append(Spacer(1, 0.1 * inch))
                    for line in lines[1:]:
                        if line.strip():
                            elements.append(Paragraph(line.strip(), styles['BulletPoint']))
                            elements.append(Spacer(1, 0.05 * inch))
                else:
                    for line in lines:
                        if line.strip():
                            elements.append(Paragraph(line.strip(), styles['Normal']))
                            elements.append(Spacer(1, 0.05 * inch))
                elements.append(Spacer(1, 0.2 * inch))
        doc.build(elements)
        print(f"✅ PDF report generated successfully: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
        fallback_filename = f"AI_Analysis_{service_id}.txt"
        with open(fallback_filename, 'w') as f:
            f.write(f"AI ANALYSIS REPORT\n")
            f.write(f"Service ID: {service_id}\n")
            f.write(f"Customer: {customer}\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            f.write(ai_analysis)
        return fallback_filename

# ========== LOAD DATA ==========
try:
    outage_df = pd.read_csv("outage_data.csv")
    crew_df = pd.read_csv("crew_data.csv")
    print("✅ Successfully loaded CSV files")
except Exception as e:
    print(f"❌ File not found: {e}")
    exit()

active_outages = outage_df[outage_df["outage status"].isin(["Pending", "Detected"])]
print(f"🔍 Found {len(active_outages)} active outages to process")

# ========== MAIN LOOP ==========
processed_count = 0
for _, outage in active_outages.iterrows():
    try:
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
                f"Crew Location: {crew['Location of crew']}\n"
                f"Distance to site: {crew['distance']:.2f} km\n"
                f"ETA: {int(crew['distance'] * 2)} minutes"
            )
            nav_map_link = google_maps_directions_link(
                comp_lat, comp_lon,
                crew['Location of crew latitude'], crew['Location of crew longitude']
            )
            osm_map_url = osm_static_map_url(
                comp_lat, comp_lon,
                crew['Location of crew latitude'], crew['Location of crew longitude']
            )
            print(f"👷 Assigned crew: {crew['Crew name']} ({crew['distance']:.2f} km away)")
        else:
            crew_assignment = "No available crew for dispatch - Manual assignment required"
            nav_map_link = gm_link
            osm_map_url = osm_static_map_url(comp_lat, comp_lon, comp_lat, comp_lon)
            print("⚠️ No available crew found for dispatch")

        nav_map_html = f"""
        <div class="section nav-section">
          <h3>🗺️ Navigation &amp; Map</h3>
          <a href="{nav_map_link}" target="_blank">
            <img src="{osm_map_url}" alt="Crew and outage locations map"
                 style="width:100%;max-width:650px;border-radius:9px;box-shadow:0 2px 6px #aaa;margin-bottom:8px;">
          </a>
          <p>
            <a href="{nav_map_link}" class="button" target="_blank"
               style="background:#1976d2;color:white;padding:9px 20px;border-radius:5px;text-decoration:none;display:inline-block;margin-right:7px;">
              🧭 Open Route in Google Maps
            </a>
          </p>
          <small>Map powered by OpenStreetMap – displays markers for crew and outage locations.</small>
        </div>
        """

        # Generate AI analysis
        print("🤖 Generating comprehensive AI analysis...")
        ai_analysis = generate_ai_analysis(
            customer, customer_id, component, comp_id,
            address, detected_time, status, crew_assignment
        )
        print("✅ AI analysis generated successfully")

        outage_details = {
            'customer': customer, 'component': component, 'address': address, 'status': status
        }
        pdf_filename = generate_pdf_report(ai_analysis, service_id, customer, outage_details)
        print(f"📄 PDF report generated: {pdf_filename}")

        # ------ Send main (user) Email -----
        sender_email = os.getenv("SMTP_EMAIL", "prathamkaushal08@gmail.com")
        sender_password = os.getenv("SMTP_PASSWORD", "dlhw kedy surr ydbr")
        recipient_email = "prathamkaushal08@gmail.com"
        subject = f"🚨 CRITICAL OUTAGE ALERT - {service_id} | {customer}"

        html_body = f"""
        <!DOCTYPE html><html><head>
        <style>
        body {{ font-family: Arial,sans-serif; background:#f2f4f8; color:#333; }}
        .header {{ background:#d32f2f; color:white; padding:26px 18px; border-radius:6px 6px 0 0; text-align:center; }}
        .content {{ background:white; border-radius:0 0 9px 9px; max-width:680px; margin:auto; box-shadow:0 2px 10px #eee; }}
        h1 {{ font-size:2em; margin:0 0 6px 0; }}
        h2 {{ margin:0 0 4px 0; font-size:1.17em; letter-spacing:2px; }}
        .section {{ margin:23px 23px 0; background:#f9fafd; padding:20px 22px; border-radius:6px; box-shadow:0 1px 4px #eee; }}
        .outage-info {{ background:#faf7f7; border-left:5px solid #d32f2f; }}
        .crew-info {{ background:#f7faf7; border-left:5px solid #43a047; }}
        .ai-section {{ background:#f2f0fa; border-left:5px solid #1976d2; }}
        .nav-section {{ background:#e3f2fd; border-left:5px solid #1976d2; }}
        .button {{ margin:5px 9px 0 0; }}
        .footer {{ color:#888; font-size:12px; text-align:center; margin:18px 0 4px 0; }}
        </style>
        </head><body>
        <div class="content">
          <div class="header">
            <h1>🚨 CRITICAL OUTAGE DETECTED</h1>
            <h2>Service ID: {service_id}</h2>
          </div>
          <div class="section outage-info">
            <h3>📍 Outage Details</h3>
            <p><b>Customer:</b> {customer} (ID: {customer_id})</p>
            <p><b>Location:</b> {address}</p>
            <p><b>Component:</b> {component} (ID: {comp_id})</p>
            <p><b>Status:</b> <span style="color:#d32f2f; font-weight:bold;">{status}</span></p>
            <p><b>Detection Time:</b> {detected_time}</p>
            <p><b>Coordinates:</b> {comp_lat}, {comp_lon}</p>
          </div>
          <div class="section crew-info">
            <h3>👷 Assigned Crew</h3>
            <pre style="font-size:13px;">{crew_assignment}</pre>
          </div>
          {nav_map_html}
          <div class="section ai-section">
            <h3>🤖 AI Analysis Preview</h3>
            <ul>
              <li>Severity Assessment</li>
              <li>Root Cause Analysis</li>
              <li>Immediate Actions Required</li>
              <li>Maintenance Recommendations</li>
              <li>Crew Deployment Strategy</li>
            </ul>
            <b>📄 Download complete analysis:</b> See attached PDF report
          </div>
          <div class="section">
            <h3>⚡ Immediate Actions Required</h3>
            <ol>
              <li>Confirm crew dispatch immediately</li>
              <li>Review attached AI analysis report</li>
              <li>Follow navigation link for optimal route</li>
              <li>Update customer on estimated restoration time</li>
            </ol>
          </div>
          <div class="footer">
            <hr>
            <div>This alert was automatically generated by the Advanced Outage Management System<br>
            Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
          </div>
        </div>
        </body></html>
        """

        msg = MIMEMultipart('alternative')
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, 'html'))
        try:
            with open(pdf_filename, 'rb') as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                pdf_attachment.add_header('Content-Disposition', 'attachment', filename=f"AI_Analysis_{service_id}.pdf")
                msg.attach(pdf_attachment)
        except Exception as e:
            print(f"⚠️ Could not attach PDF: {e}")

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"✅ Enhanced user email sent for outage {service_id}")
            processed_count += 1
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
        except Exception as e:
            print(f"❌ Failed to send email for outage {service_id}: {str(e)}")

        # ------- CREW EMAIL ---------
        crew_recipient_email = "eaglekau08@gmail.com"    # Set to actual crew email if available
        crew_subject = f"🛠️ New Outage Assignment: {service_id} | {customer}"

        crew_html_body = f"""
        <!DOCTYPE html><html><head>
        <style>
        body {{ font-family: Arial,sans-serif; background:#f4f7fc; color:#222; }}
        .header {{ background:#1976d2; color:white; padding:23px 18px; border-radius:6px 6px 0 0; text-align:center; }}
        .content {{ background:white; border-radius:0 0 9px 9px; max-width:680px; margin:auto; box-shadow:0 2px 6px #eee; }}
        h1 {{ font-size:1.6em; margin:0 0 8px 0; }}
        h2 {{ margin:0 0 3px 0; font-size:1.12em; letter-spacing:2px; }}
        .section {{ margin:22px 22px 0; background:#f5faff; padding:18px 22px; border-radius:6px; box-shadow:0 1px 4px #eee; }}
        .nav-section {{ background:#e2f3fa; border-left:5px solid #1976d2; }}
        .button {{ margin:5px 8px 0 0; }}
        .footer {{ color:#666; font-size:12px; text-align:center; margin:18px 0 4px 0; }}
        </style>
        </head><body>
        <div class="content">
          <div class="header">
            <h1>🛠️ Outage Assignment - Service ID: {service_id}</h1>
          </div>
          <div class="section">
            <h3>📋 Customer Details</h3>
            <p><b>Name:</b> {customer}</p>
            <p><b>Address:</b> {address}</p>
            <p><b>Component:</b> {component} (ID: {comp_id})</p>
            <p><b>Status:</b> {status}</p>
            <p><b>Detection Time:</b> {detected_time}</p>
          </div>
          {nav_map_html}
          <div class="section">
            <h3>👷 Assigned To:</h3>
            <pre style="font-size:13px;">{crew_assignment}</pre>
          </div>
          <div class="footer">
            <hr>
            <div>This assignment was automatically generated by the Advanced Outage Management System<br>
            Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
          </div>
        </div>
        </body></html>
        """

        crew_msg = MIMEMultipart('alternative')
        crew_msg["From"] = sender_email
        crew_msg["To"] = crew_recipient_email
        crew_msg["Subject"] = crew_subject
        crew_msg.attach(MIMEText(crew_html_body, 'html'))
        try:
            with open(pdf_filename, 'rb') as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
                pdf_attachment.add_header('Content-Disposition', 'attachment', filename=f"AI_Analysis_{service_id}.pdf")
                crew_msg.attach(pdf_attachment)
        except Exception as e:
            print(f"⚠️ Could not attach PDF to crew email: {e}")
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(crew_msg)
            print(f"✅ Crew email sent for outage {service_id}")
        except Exception as e:
            print(f"❌ Failed to send crew email for outage {service_id}: {str(e)}")

    except Exception as e:
        print(f"❌ Error processing outage: {e}")

print(f"\n🎯 Processing complete! {processed_count} outages processed and enhanced alerts sent.")
