import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Mail, MapPin } from "lucide-react";
import { SiLinkedin } from "react-icons/si";
import { useToast } from "@/hooks/use-toast";
import { ResumePreviewCard } from "@/components/resume-download";

export default function ContactSection() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: ""
  });
  const { toast } = useToast();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In a real implementation, this would send the form data to a backend
    toast({
      title: "Message Sent!",
      description: "Thank you for your message. I'll get back to you soon.",
    });
    setFormData({ name: "", email: "", message: "" });
  };

  return (
    <section id="contact" className="py-20 bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4 text-foreground" data-testid="contact-title">
            Get In Touch
          </h2>
          <div className="w-24 h-1 bg-primary mx-auto rounded-full mb-6"></div>
          <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="contact-description">
            Let's discuss opportunities to collaborate on data science and machine learning projects
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          <div className="lg:col-span-2">
            <Card className="border border-border" data-testid="contact-card">
              <CardContent className="p-8 md:p-12">
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                <h3 className="text-2xl font-semibold mb-6 text-foreground" data-testid="contact-info-title">
                  Contact Information
                </h3>
                <div className="space-y-6">
                  <div className="flex items-center space-x-4" data-testid="contact-email">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                      <Mail className="text-primary h-5 w-5" />
                    </div>
                    <div>
                      <h4 className="font-medium text-foreground">Email</h4>
                      <a 
                        href="mailto:datascienceai50@gmail.com" 
                        className="text-muted-foreground hover:text-primary transition-colors"
                        data-testid="link-email"
                      >
                        datascienceai50@gmail.com
                      </a>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4" data-testid="contact-linkedin">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                      <SiLinkedin className="text-primary text-xl" />
                    </div>
                    <div>
                      <h4 className="font-medium text-foreground">LinkedIn</h4>
                      <a 
                        href="https://linkedin.com/in/prashant-yadav-a2a868277" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-primary transition-colors"
                        data-testid="link-linkedin"
                      >
                        linkedin.com/in/prashant-yadav-a2a868277
                      </a>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4" data-testid="contact-location">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                      <MapPin className="text-primary h-5 w-5" />
                    </div>
                    <div>
                      <h4 className="font-medium text-foreground">Location</h4>
                      <p className="text-muted-foreground">Mathura, UP, India</p>
                    </div>
                  </div>
                </div>
                  </div>

                  <div>
                <h3 className="text-2xl font-semibold mb-6 text-foreground" data-testid="contact-form-title">
                  Quick Message
                </h3>
                <form onSubmit={handleSubmit} className="space-y-4" data-testid="contact-form">
                  <Input
                    type="text"
                    name="name"
                    placeholder="Your Name"
                    value={formData.name}
                    onChange={handleInputChange}
                    required
                    data-testid="input-name"
                  />
                  <Input
                    type="email"
                    name="email"
                    placeholder="Your Email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required
                    data-testid="input-email"
                  />
                  <Textarea
                    name="message"
                    placeholder="Your Message"
                    rows={4}
                    value={formData.message}
                    onChange={handleInputChange}
                    required
                    className="resize-none"
                    data-testid="textarea-message"
                  />
                  <Button 
                    type="submit" 
                    className="w-full"
                    data-testid="button-send-message"
                  >
                    Send Message
                  </Button>
                </form>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
          
          <div>
            <ResumePreviewCard />
          </div>
        </div>
      </div>
    </section>
  );
}
