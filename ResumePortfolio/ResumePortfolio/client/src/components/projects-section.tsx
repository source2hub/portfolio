import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, Filter, X } from "lucide-react";
import { Link } from "wouter";
import { projects, getAllCategories } from "@/data/projects";

export default function ProjectsSection() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedTech, setSelectedTech] = useState<string | null>(null);
  
  const categories = getAllCategories();
  const allTechnologies = Array.from(new Set(projects.flatMap(project => project.technologies)));
  
  const filteredProjects = projects.filter(project => {
    const matchesCategory = !selectedCategory || project.category === selectedCategory;
    const matchesTech = !selectedTech || project.technologies.includes(selectedTech);
    return matchesCategory && matchesTech;
  });

  const clearFilters = () => {
    setSelectedCategory(null);
    setSelectedTech(null);
  };

  return (
    <section id="projects" className="py-20 bg-background">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4 text-foreground" data-testid="projects-title">
            Featured Projects
          </h2>
          <div className="w-24 h-1 bg-primary mx-auto rounded-full mb-6"></div>
          <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="projects-description">
            Showcase of my data science and machine learning projects, demonstrating expertise across various domains
          </p>
        </div>

        {/* Filters */}
        <div className="mb-12" data-testid="project-filters">
          <div className="flex items-center justify-center mb-6">
            <Filter className="h-5 w-5 text-primary mr-2" />
            <h3 className="text-lg font-semibold text-foreground">Filter Projects</h3>
          </div>
          
          <div className="flex flex-wrap justify-center gap-3 mb-6">
            <div className="text-sm font-medium text-muted-foreground mr-2">Category:</div>
            {categories.map((category) => (
              <Button
                key={category}
                variant={selectedCategory === category ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedCategory(selectedCategory === category ? null : category)}
                className="text-xs"
                data-testid={`filter-category-${category.toLowerCase().replace(/\s+/g, '-')}`}
              >
                {category}
              </Button>
            ))}
          </div>

          <div className="flex flex-wrap justify-center gap-2 mb-6">
            <div className="text-sm font-medium text-muted-foreground mr-2">Technology:</div>
            {allTechnologies.slice(0, 10).map((tech) => (
              <Button
                key={tech}
                variant={selectedTech === tech ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedTech(selectedTech === tech ? null : tech)}
                className="text-xs"
                data-testid={`filter-tech-${tech.toLowerCase().replace(/\s+/g, '-')}`}
              >
                {tech}
              </Button>
            ))}
          </div>

          {(selectedCategory || selectedTech) && (
            <div className="flex justify-center mb-6">
              <Button
                variant="ghost"
                size="sm"
                onClick={clearFilters}
                className="text-muted-foreground hover:text-foreground"
                data-testid="clear-filters"
              >
                <X className="h-4 w-4 mr-1" />
                Clear Filters ({filteredProjects.length} project{filteredProjects.length !== 1 ? 's' : ''})
              </Button>
            </div>
          )}
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredProjects.map((project, index) => (
            <Card 
              key={index} 
              className={`project-card border border-border overflow-hidden ${index === 4 ? 'md:col-span-2 lg:col-span-1' : ''}`}
              data-testid={project.testId}
            >
              <img
                src={project.image}
                alt={project.alt}
                className="w-full h-48 object-cover"
                data-testid={`${project.testId}-image`}
              />
              <CardContent className="p-6">
                <h3 className="text-xl font-semibold mb-3 text-foreground" data-testid={`${project.testId}-title`}>
                  {project.title}
                </h3>
                <p className="text-muted-foreground mb-4 text-sm leading-relaxed" data-testid={`${project.testId}-description`}>
                  {project.description}
                </p>
                <div className="flex flex-wrap gap-2 mb-4" data-testid={`${project.testId}-technologies`}>
                  {project.technologies.map((tech, techIndex) => (
                    <Badge
                      key={techIndex}
                      variant="secondary"
                      className="skill-tag bg-primary/10 text-primary px-3 py-1 rounded-full text-xs font-medium"
                    >
                      {tech}
                    </Badge>
                  ))}
                </div>
                <Link href={`/project/${project.id}`}>
                  <button 
                    className="text-primary hover:text-primary/80 font-medium text-sm flex items-center transition-colors"
                    data-testid={`${project.testId}-link`}
                  >
                    View Project <ExternalLink className="ml-1 h-3 w-3" />
                  </button>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
