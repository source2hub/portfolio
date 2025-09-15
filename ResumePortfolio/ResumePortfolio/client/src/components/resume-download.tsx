import { Download, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { resumeData } from "@/data/resume-data";

interface ResumeDownloadProps {
  variant?: "default" | "outline" | "ghost";
  size?: "sm" | "default" | "lg";
  className?: string;
  showIcon?: boolean;
}

export default function ResumeDownload({ 
  variant = "default", 
  size = "default", 
  className = "",
  showIcon = true 
}: ResumeDownloadProps) {
  const { toast } = useToast();

  const generateResumeHTML = () => {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${resumeData.personalInfo.name} - Resume</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Arial', sans-serif; 
            line-height: 1.6; 
            color: #333; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: white;
        }
        .header { 
            text-align: center; 
            border-bottom: 3px solid #3b82f6; 
            padding-bottom: 20px; 
            margin-bottom: 30px; 
        }
        .name { 
            font-size: 36px; 
            font-weight: bold; 
            color: #1e40af; 
            margin-bottom: 8px; 
        }
        .title { 
            font-size: 18px; 
            color: #6b7280; 
            margin-bottom: 15px; 
        }
        .contact-info { 
            font-size: 14px; 
            color: #4b5563; 
        }
        .contact-info span { 
            margin: 0 10px; 
        }
        .section { 
            margin-bottom: 30px; 
        }
        .section-title { 
            font-size: 20px; 
            font-weight: bold; 
            color: #1e40af; 
            border-bottom: 2px solid #e5e7eb; 
            padding-bottom: 8px; 
            margin-bottom: 15px; 
        }
        .summary { 
            font-size: 16px; 
            line-height: 1.7; 
            color: #374151; 
            text-align: justify; 
        }
        .skills-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
        }
        .skill-category { 
            background: #f8fafc; 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #3b82f6; 
        }
        .skill-category-title { 
            font-weight: bold; 
            color: #1e40af; 
            margin-bottom: 8px; 
        }
        .skill-items { 
            font-size: 14px; 
            color: #4b5563; 
        }
        .project { 
            margin-bottom: 20px; 
            padding: 15px; 
            background: #fafafa; 
            border-radius: 8px; 
            border-left: 4px solid #10b981; 
        }
        .project-title { 
            font-weight: bold; 
            color: #059669; 
            margin-bottom: 8px; 
            font-size: 16px; 
        }
        .project-description { 
            color: #374151; 
            margin-bottom: 10px; 
            line-height: 1.6; 
        }
        .project-technologies { 
            font-size: 12px; 
            color: #6b7280; 
        }
        .tech-tag { 
            background: #dbeafe; 
            color: #1e40af; 
            padding: 2px 8px; 
            border-radius: 12px; 
            margin-right: 5px; 
            display: inline-block; 
            margin-bottom: 5px; 
        }
        .education, .certifications { 
            background: #f1f5f9; 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #f59e0b; 
        }
        .education-title, .cert-title { 
            font-weight: bold; 
            color: #d97706; 
        }
        .education-details, .cert-details { 
            color: #4b5563; 
            margin-top: 5px; 
        }
        .cert-item { 
            margin-bottom: 10px; 
        }
        @media print {
            body { 
                padding: 0; 
                font-size: 12px; 
            }
            .name { 
                font-size: 28px; 
            }
            .section { 
                margin-bottom: 20px; 
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="name">${resumeData.personalInfo.name}</div>
        <div class="title">${resumeData.personalInfo.title}</div>
        <div class="contact-info">
            <span>📧 ${resumeData.personalInfo.email}</span>
            <span>🔗 ${resumeData.personalInfo.linkedin}</span>
            <span>📍 ${resumeData.personalInfo.location}</span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Professional Summary</div>
        <div class="summary">${resumeData.summary}</div>
    </div>

    <div class="section">
        <div class="section-title">Technical Skills</div>
        <div class="skills-grid">
            ${resumeData.skills.map(skillCategory => `
                <div class="skill-category">
                    <div class="skill-category-title">${skillCategory.category}</div>
                    <div class="skill-items">${skillCategory.items.join(', ')}</div>
                </div>
            `).join('')}
        </div>
    </div>

    <div class="section">
        <div class="section-title">Featured Projects</div>
        ${resumeData.projects.map(project => `
            <div class="project">
                <div class="project-title">${project.title}</div>
                <div class="project-description">${project.description}</div>
                <div class="project-technologies">
                    ${project.technologies.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                </div>
            </div>
        `).join('')}
    </div>

    <div class="section">
        <div class="section-title">Education</div>
        <div class="education">
            <div class="education-title">${resumeData.education.degree}</div>
            <div class="education-details">${resumeData.education.institution} • ${resumeData.education.year}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Certifications</div>
        <div class="certifications">
            ${resumeData.certifications.map(cert => `
                <div class="cert-item">
                    <div class="cert-title">${cert.title}</div>
                    <div class="cert-details">${cert.provider}</div>
                </div>
            `).join('')}
        </div>
    </div>
</body>
</html>`;
  };

  const downloadResume = () => {
    try {
      const htmlContent = generateResumeHTML();
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = url;
      link.download = `${resumeData.personalInfo.name.replace(/\s+/g, '_')}_Resume.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast({
        title: "Resume Downloaded!",
        description: "Your resume has been downloaded as an HTML file. You can open it in any browser or convert it to PDF.",
      });
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "There was an error downloading your resume. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <Button
      onClick={downloadResume}
      variant={variant}
      size={size}
      className={className}
      data-testid="button-download-resume"
    >
      {showIcon && <Download className="mr-2 h-4 w-4" />}
      Download Resume
    </Button>
  );
}

export function ResumePreviewCard() {
  return (
    <div className="bg-card border border-border rounded-lg p-6" data-testid="resume-preview-card">
      <div className="flex items-center mb-4">
        <FileText className="h-6 w-6 text-primary mr-3" />
        <h3 className="text-xl font-semibold text-foreground">Professional Resume</h3>
      </div>
      <p className="text-muted-foreground mb-4">
        Download a comprehensive resume showcasing my AI/ML expertise, projects, and professional experience.
      </p>
      <div className="space-y-2 text-sm text-muted-foreground mb-4">
        <div>✓ Complete project portfolio with technical details</div>
        <div>✓ Comprehensive skills and certifications</div>
        <div>✓ Professional formatting optimized for both screen and print</div>
        <div>✓ Ready for HR systems and applicant tracking software</div>
      </div>
      <ResumeDownload variant="default" className="w-full" />
    </div>
  );
}