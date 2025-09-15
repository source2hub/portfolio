import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Code2, 
  Brain, 
  Database, 
  BarChart3, 
  Languages, 
  Bot 
} from "lucide-react";
import { SiPython } from "react-icons/si";

interface SkillCategory {
  title: string;
  icon: React.ReactNode;
  skills: string[];
  testId: string;
}

const skillCategories: SkillCategory[] = [
  {
    title: "Python Stack",
    icon: <SiPython className="text-2xl" />,
    skills: ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib"],
    testId: "skills-python"
  },
  {
    title: "Machine Learning",
    icon: <Bot className="h-6 w-6" />,
    skills: ["ML Algorithms", "Statistics", "Feature Engineering", "Model Selection"],
    testId: "skills-ml"
  },
  {
    title: "Deep Learning",
    icon: <Brain className="h-6 w-6" />,
    skills: ["TensorFlow", "PyTorch", "CNN", "Neural Networks"],
    testId: "skills-dl"
  },
  {
    title: "Data Engineering",
    icon: <Database className="h-6 w-6" />,
    skills: ["SQL", "Web Scraping", "ETL", "Data Pipelines"],
    testId: "skills-data-eng"
  },
  {
    title: "Data Visualization",
    icon: <BarChart3 className="h-6 w-6" />,
    skills: ["Plotly", "Seaborn", "Matplotlib", "Dashboard"],
    testId: "skills-visualization"
  },
  {
    title: "NLP & Text Analytics",
    icon: <Languages className="h-6 w-6" />,
    skills: ["NLP", "Vectorization", "Text Similarity", "Sentiment Analysis"],
    testId: "skills-nlp"
  }
];

export default function SkillsSection() {
  return (
    <section id="skills" className="py-20 bg-muted/30">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4 text-foreground" data-testid="skills-title">
            Technical Skills
          </h2>
          <div className="w-24 h-1 bg-primary mx-auto rounded-full mb-6"></div>
          <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="skills-description">
            Comprehensive expertise across the data science and machine learning technology stack
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {skillCategories.map((category, index) => (
            <Card key={index} className="border border-border" data-testid={category.testId}>
              <CardContent className="p-8">
                <div className="flex items-center mb-6">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mr-4 text-primary">
                    {category.icon}
                  </div>
                  <h3 className="text-xl font-semibold text-foreground" data-testid={`${category.testId}-title`}>
                    {category.title}
                  </h3>
                </div>
                <div className="flex flex-wrap gap-2" data-testid={`${category.testId}-skills`}>
                  {category.skills.map((skill, skillIndex) => (
                    <Badge
                      key={skillIndex}
                      variant="secondary"
                      className="skill-tag bg-primary/10 text-primary px-3 py-2 rounded-lg text-sm font-medium"
                    >
                      {skill}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
