"""
PDF Report Builder Module

This module builds comprehensive PDF reports for pruning analysis.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class PDFReportBuilder:
    """
    Builds PDF reports using ReportLab.
    """
    
    def __init__(
        self,
        filename: str,
        title: str = "Neural Network Pruning Report",
        author: str = "CleanAI Framework",
        pagesize = A4
    ):
        """
        Initialize PDF builder.
        
        Args:
            filename: Output PDF filename
            title: Report title
            author: Report author
            pagesize: Page size (default A4)
        """
        self.filename = filename
        self.title = title
        self.author = author
        self.pagesize = pagesize
        
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=pagesize,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch,
        )
        
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#e74c3c'),
            leftIndent=20,
            fontName='Helvetica-Bold'
        ))
        
        # Success style
        self.styles.add(ParagraphStyle(
            name='Success',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#27ae60'),
            leftIndent=20,
            fontName='Helvetica-Bold'
        ))
    
    def add_cover_page(
        self,
        model_name: str,
        model_type: str,
        dataset: str,
        pruning_ratio: float
    ):
        """
        Add cover page.
        
        Args:
            model_name: Model name
            model_type: Model type
            dataset: Dataset name
            pruning_ratio: Pruning ratio
        """
        # Title
        self.story.append(Spacer(1, 2*inch))
        title = Paragraph(self.title, self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_CENTER
        )
        subtitle = Paragraph(
            f"<b>Model:</b> {model_name} | <b>Type:</b> {model_type}<br/>"
            f"<b>Dataset:</b> {dataset} | <b>Pruning:</b> {pruning_ratio:.1f}%",
            subtitle_style
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 1*inch))
        
        # Date
        date_text = Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            subtitle_style
        )
        self.story.append(date_text)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Framework
        framework_text = Paragraph(
            f"<b>Framework:</b> CleanAI v0.1.0",
            subtitle_style
        )
        self.story.append(framework_text)
        
        self.story.append(PageBreak())
    
    def add_executive_summary(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ):
        """
        Add executive summary section.
        
        Args:
            before: Before-pruning metrics
            after: After-pruning metrics
        """
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Summary text
        summary_text = f"""
        This report presents a comprehensive analysis of neural network pruning operations.
        The model was successfully pruned with a reduction of {after.get('param_reduction_pct', 0):.1f}% 
        in parameters while maintaining {after.get('accuracy', 0):.2f}% accuracy.
        """
        self.story.append(Paragraph(summary_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Comparison table
        data = [
            ['Metric', 'Before Pruning', 'After Pruning', 'Change'],
            [
                'Parameters',
                f"{before.get('parameters', 0):,}",
                f"{after.get('parameters', 0):,}",
                f"-{after.get('param_reduction_pct', 0):.1f}%"
            ],
            [
                'Model Size (MB)',
                f"{before.get('model_size_mb', 0):.2f}",
                f"{after.get('model_size_mb', 0):.2f}",
                f"-{after.get('size_reduction_pct', 0):.1f}%"
            ],
            [
                'GFLOPs',
                f"{before.get('gflops', 0):.2f}",
                f"{after.get('gflops', 0):.2f}",
                f"-{after.get('flops_reduction_pct', 0):.1f}%"
            ],
            [
                'Inference Time (ms)',
                f"{before.get('inference_time_ms', 0):.2f}",
                f"{after.get('inference_time_ms', 0):.2f}",
                f"{after.get('speedup', 1):.2f}x faster"
            ],
            [
                'Accuracy (%)',
                f"{before.get('accuracy', 0):.2f}",
                f"{after.get('accuracy', 0):.2f}",
                f"-{after.get('accuracy_drop', 0):.2f}%"
            ],
        ]
        
        table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_section(self, title: str):
        """Add section header."""
        self.story.append(PageBreak())
        self.story.append(Paragraph(title, self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_subsection(self, title: str):
        """Add subsection header."""
        self.story.append(Paragraph(title, self.styles['SubSection']))
        self.story.append(Spacer(1, 0.1*inch))
    
    def add_text(self, text: str, style_name: str = 'Normal'):
        """Add text paragraph."""
        self.story.append(Paragraph(text, self.styles[style_name]))
        self.story.append(Spacer(1, 0.1*inch))
    
    def add_table(self, data: List[List[str]], col_widths: Optional[List[float]] = None):
        """Add table."""
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_image(self, image_path_or_obj, width: float = 6*inch, height: Optional[float] = None):
        """Add image."""
        if isinstance(image_path_or_obj, str):
            img = RLImage(image_path_or_obj, width=width, height=height)
        else:
            # PIL Image - save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                image_path_or_obj.save(tmp.name, format='PNG')
                img = RLImage(tmp.name, width=width, height=height)
        
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_risk_warnings(self, risks: List[Dict[str, str]]):
        """
        Add risk warnings section.
        
        Args:
            risks: List of risk dictionaries
        """
        self.add_subsection("Risk Analysis")
        
        if not risks:
            self.add_text("✓ No significant risks detected.", 'Success')
        else:
            for risk in risks:
                level = risk['level']
                category = risk['category']
                message = risk['message']
                
                color = '#e74c3c' if level == 'HIGH' else '#f39c12'
                icon = '⚠️' if level == 'HIGH' else '⚡'
                
                risk_style = ParagraphStyle(
                    'Risk',
                    parent=self.styles['Normal'],
                    fontSize=10,
                    textColor=colors.HexColor(color),
                    leftIndent=20,
                    spaceBefore=5,
                    spaceAfter=5
                )
                
                text = f"{icon} <b>[{level}] {category}:</b> {message}"
                self.story.append(Paragraph(text, risk_style))
        
        self.story.append(Spacer(1, 0.2*inch))
    
    def build(self):
        """Build the PDF document."""
        self.doc.build(self.story)
        print(f"\n✅ Report generated: {self.filename}")
