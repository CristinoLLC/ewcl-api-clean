from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from typing import List, Dict, Any

def create_pdf_report(analysis_data: Dict[str, Any]) -> bytes:
    """
    Generate PDF report from EWCL analysis data
    
    Args:
        analysis_data: Analysis results with scores, metrics, and configuration
    
    Returns:
        PDF as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    warning_style = ParagraphStyle(
        'Warning',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.red,
        spaceAfter=12
    )
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=12
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("EWCL Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Raw entropy warning if enabled
    use_raw_ewcl = analysis_data.get("use_raw_ewcl", False)
    if use_raw_ewcl:
        elements.append(Paragraph(
            "⚠️ Raw Entropy View Enabled – EWCL scores shown are unnormalized entropy magnitudes.",
            warning_style
        ))
    
    # Analysis metadata
    metadata_text = f"""
    Model: {analysis_data.get('model', 'N/A')}<br/>
    Lambda: {analysis_data.get('lambda', 'N/A')}<br/>
    Normalized: {analysis_data.get('normalized', 'N/A')}<br/>
    Raw EWCL Mode: {use_raw_ewcl}<br/>
    Total Residues: {analysis_data.get('n_residues', 'N/A')}<br/>
    Generated: {analysis_data.get('generated', 'N/A')}
    """
    elements.append(Paragraph(metadata_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Metrics summary
    metrics = analysis_data.get('metrics', {})
    if metrics:
        elements.append(Paragraph("Analysis Metrics", styles['Heading2']))
        metrics_text = f"""
        Pearson Correlation: {metrics.get('pearson', 'N/A')}<br/>
        Spearman Correlation: {metrics.get('spearman', 'N/A')}<br/>
        Local Spearman Avg: {metrics.get('spearman_local_avg', 'N/A')}<br/>
        AUC (Pseudo-pLDDT): {metrics.get('auc_pseudo_plddt', 'N/A')}<br/>
        Kendall Tau: {metrics.get('kendall_tau', 'N/A')}<br/>
        Mismatches: {metrics.get('n_mismatches', 'N/A')}
        """
        elements.append(Paragraph(metrics_text, styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Residue scores table
    results = analysis_data.get('results', [])
    if results:
        elements.append(Paragraph("Residue Scores", styles['Heading2']))
        
        # Table headers - dual columns for normalized and raw scores
        table_data = [["Residue", "Normalized EWCL", "Raw EWCL", "pLDDT", "B-factor"]]
        
        # Add data rows
        for r in results[:50]:  # Limit to first 50 residues for PDF readability
            table_data.append([
                r.get("residue_id", "N/A"),
                round(r.get("cl", 0), 3),
                round(r.get("raw_cl", 0), 2),
                round(r.get("plddt", 0), 1),
                round(r.get("b_factor", 0), 1)
            ])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        if len(results) > 50:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Note: Showing first 50 of {len(results)} residues", styles['Italic']))
    
    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
