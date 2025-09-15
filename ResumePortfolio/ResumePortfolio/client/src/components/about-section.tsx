import { Card, CardContent } from "@/components/ui/card";
import { GraduationCap, MapPin, Award, Brain, Database } from "lucide-react";

export default function AboutSection() {
  return (
    <section id="about" className="py-20 bg-muted/30">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4 text-foreground" data-testid="about-title">
            About Me
          </h2>
          <div className="w-24 h-1 bg-primary mx-auto rounded-full"></div>
        </div>

        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h3 className="text-2xl font-semibold mb-6 text-foreground" data-testid="professional-overview-title">
              Professional Overview
            </h3>
            <p className="text-muted-foreground leading-relaxed mb-6" data-testid="about-description-1">
              Accomplished Data Scientist with expertise in compiling, transforming, and analyzing complex information through advanced software solutions. I specialize in machine learning algorithms, deep learning architectures, and large-scale dataset management.
            </p>
            <p className="text-muted-foreground leading-relaxed mb-8" data-testid="about-description-2">
              My demonstrated success includes identifying meaningful relationships in data and building innovative solutions to complex business problems. I'm passionate about leveraging AI and ML to drive data-driven decision making.
            </p>

            <div className="grid grid-cols-2 gap-6">
              <Card className="border border-border" data-testid="education-card">
                <CardContent className="p-6">
                  <div className="flex items-center mb-2">
                    <GraduationCap className="h-5 w-5 text-primary mr-2" />
                    <h4 className="font-semibold text-primary">Education</h4>
                  </div>
                  <p className="text-sm text-muted-foreground">Mechanical Engineer</p>
                  <p className="text-sm text-muted-foreground">Dr. A.P.J. Abdul Kalam Technical University</p>
                  <p className="text-sm text-muted-foreground">2014</p>
                </CardContent>
              </Card>
              <Card className="border border-border" data-testid="location-card">
                <CardContent className="p-6">
                  <div className="flex items-center mb-2">
                    <MapPin className="h-5 w-5 text-primary mr-2" />
                    <h4 className="font-semibold text-primary">Location</h4>
                  </div>
                  <p className="text-sm text-muted-foreground">Mathura, UP, India</p>
                  <p className="text-sm text-muted-foreground mt-2">Open to Remote Work</p>
                </CardContent>
              </Card>
            </div>
          </div>

          <Card className="border border-border" data-testid="certifications-card">
            <CardContent className="p-8">
              <h3 className="text-xl font-semibold mb-6 text-foreground">Certifications</h3>
              <div className="space-y-4">
                <div className="flex items-center space-x-4" data-testid="cert-python-ml">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Brain className="text-primary h-6 w-6" />
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">Python & Machine Learning</h4>
                    <p className="text-sm text-muted-foreground">CloudyML</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4" data-testid="cert-deep-learning">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Award className="text-primary h-6 w-6" />
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">Deep Learning</h4>
                    <p className="text-sm text-muted-foreground">CloudyML</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4" data-testid="cert-data-science">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Database className="text-primary h-6 w-6" />
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">Data Science</h4>
                    <p className="text-sm text-muted-foreground">Oracle</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
