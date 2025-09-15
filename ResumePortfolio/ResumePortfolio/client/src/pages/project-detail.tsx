import { useParams, Link } from "wouter";
import { ArrowLeft, ExternalLink, Github, Play, ChevronRight, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { getProjectById } from "@/data/projects";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import OlympicsVisualizations from "@/components/olympics-visualizations";

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const project = getProjectById(id || "");

  if (!project) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-foreground mb-4">Project Not Found</h1>
          <p className="text-muted-foreground mb-8">The project you're looking for doesn't exist.</p>
          <Link href="/#projects">
            <Button variant="default" data-testid="button-back-projects">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Projects
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <Link href="/#projects">
              <Button variant="ghost" size="sm" data-testid="button-back">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Projects
              </Button>
            </Link>
            <Badge variant="secondary" className="bg-primary/10 text-primary">
              {project.category}
            </Badge>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="relative">
        <div className="h-64 md:h-80 overflow-hidden">
          <img
            src={project.image}
            alt={project.alt}
            className="w-full h-full object-cover"
            data-testid="project-hero-image"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent"></div>
        </div>
        <div className="absolute bottom-0 left-0 right-0 p-6 md:p-8">
          <div className="max-w-6xl mx-auto">
            <h1 className="text-3xl md:text-5xl font-bold text-white mb-4" data-testid="project-title">
              {project.title}
            </h1>
            <p className="text-lg md:text-xl text-white/90 max-w-3xl" data-testid="project-description">
              {project.description}
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid lg:grid-cols-3 gap-12">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-12">
            {/* Overview */}
            <section data-testid="project-overview">
              <h2 className="text-3xl font-bold text-foreground mb-6">Project Overview</h2>
              <p className="text-muted-foreground leading-relaxed text-lg">
                {project.fullDescription}
              </p>
            </section>

            {/* Technical Specifications */}
            <section data-testid="project-technical-specs">
              <h2 className="text-3xl font-bold text-foreground mb-6">Technical Specifications</h2>
              <div className="grid gap-4">
                {project.technicalSpecs.map((spec, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <ChevronRight className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-muted-foreground">{spec}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Code Snippets */}
            <section data-testid="project-code-snippets">
              <h2 className="text-3xl font-bold text-foreground mb-6">Code Snippets</h2>
              <div className="space-y-8">
                {project.codeSnippets.map((snippet, index) => (
                  <Card key={index} className="border border-border">
                    <CardHeader>
                      <CardTitle className="text-xl text-foreground">{snippet.title}</CardTitle>
                      <p className="text-muted-foreground">{snippet.description}</p>
                    </CardHeader>
                    <CardContent>
                      <div className="relative">
                        <div className="absolute top-3 right-3 flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {snippet.language}
                          </Badge>
                        </div>
                        <SyntaxHighlighter
                          language={snippet.language}
                          style={tomorrow}
                          className="rounded-lg text-sm"
                          showLineNumbers
                          data-testid={`code-snippet-${index}`}
                        >
                          {snippet.code}
                        </SyntaxHighlighter>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>

            {/* Challenges & Solutions */}
            <section data-testid="project-challenges">
              <h2 className="text-3xl font-bold text-foreground mb-6">Challenges & Solutions</h2>
              <div className="space-y-4">
                {project.challenges.map((challenge, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-destructive/10 text-destructive rounded-full flex items-center justify-center text-sm font-bold mt-0.5">
                      {index + 1}
                    </div>
                    <p className="text-muted-foreground">{challenge}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Interactive Visualizations for Olympics Analysis */}
            {project.id === 'olympics-analysis' && (
              <section data-testid="project-visualizations">
                <div className="flex items-center mb-6">
                  <BarChart3 className="h-6 w-6 text-primary mr-3" />
                  <h2 className="text-3xl font-bold text-foreground">Interactive Data Visualizations</h2>
                </div>
                <OlympicsVisualizations />
              </section>
            )}

            {/* Results & Impact */}
            <section data-testid="project-results">
              <h2 className="text-3xl font-bold text-foreground mb-6">Results & Impact</h2>
              <div className="space-y-4">
                {project.results.map((result, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-green-500/10 text-green-600 rounded-full flex items-center justify-center text-sm font-bold mt-0.5">
                      ✓
                    </div>
                    <p className="text-muted-foreground">{result}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Project Details */}
            <Card className="border border-border" data-testid="project-details-card">
              <CardHeader>
                <CardTitle>Project Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h4 className="font-semibold text-foreground mb-3">Technologies Used</h4>
                  <div className="flex flex-wrap gap-2">
                    {project.technologies.map((tech, index) => (
                      <Badge key={index} variant="secondary" className="bg-primary/10 text-primary">
                        {tech}
                      </Badge>
                    ))}
                  </div>
                </div>

                <Separator />

                <div>
                  <h4 className="font-semibold text-foreground mb-3">Category</h4>
                  <Badge variant="outline" className="text-sm">
                    {project.category}
                  </Badge>
                </div>

                <Separator />

                <div className="space-y-3">
                  <h4 className="font-semibold text-foreground">Project Links</h4>
                  {project.liveDemo && (
                    <Button variant="outline" className="w-full justify-start" data-testid="button-live-demo">
                      <Play className="mr-2 h-4 w-4" />
                      Live Demo
                    </Button>
                  )}
                  {project.github && (
                    <Button variant="outline" className="w-full justify-start" data-testid="button-github">
                      <Github className="mr-2 h-4 w-4" />
                      Source Code
                    </Button>
                  )}
                  <Button variant="outline" className="w-full justify-start" data-testid="button-case-study">
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Full Case Study
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Contact CTA */}
            <Card className="border border-border bg-primary/5" data-testid="contact-cta-card">
              <CardContent className="p-6">
                <h3 className="text-xl font-semibold text-foreground mb-3">Interested in this project?</h3>
                <p className="text-muted-foreground mb-4">
                  Let's discuss how similar solutions can benefit your business.
                </p>
                <Link href="/#contact">
                  <Button className="w-full" data-testid="button-contact-about-project">
                    Get In Touch
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}