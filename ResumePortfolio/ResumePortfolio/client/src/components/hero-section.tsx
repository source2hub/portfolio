import { Button } from "@/components/ui/button";
import userPhoto from "@assets/IMG-20220303-WA0037-01_1757857133017.jpeg";

export default function HeroSection() {
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <section id="home" className="pt-16 min-h-screen flex items-center gradient-bg">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="text-white">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight" data-testid="hero-name">
              Prashant Kr.
              <span className="block text-4xl md:text-5xl font-light">Yadav</span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100" data-testid="hero-title">
              AI/ML Engineer & Data Scientist
            </p>
            <p className="text-lg mb-10 text-blue-100 leading-relaxed max-w-lg" data-testid="hero-description">
              Meticulous Data Scientist with expertise in machine learning, deep learning, and large dataset management. Passionate about transforming complex data into actionable business insights.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                onClick={() => scrollToSection("projects")}
                className="bg-white text-primary px-8 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
                data-testid="button-view-projects"
              >
                View Projects
              </Button>
              <Button
                variant="outline"
                onClick={() => scrollToSection("contact")}
                className="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-primary transition-colors bg-transparent"
                data-testid="button-get-in-touch"
              >
                Get In Touch
              </Button>
            </div>
          </div>
          <div className="flex justify-center md:justify-end">
            <div className="relative">
              {/* Enhanced background elements */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-400/20 via-purple-500/20 to-cyan-400/20 rounded-3xl blur-3xl scale-110"></div>
              <div className="absolute inset-0 bg-gradient-to-tl from-white/5 via-transparent to-white/10 rounded-3xl"></div>
              
              <div className="relative w-80 h-80 rounded-2xl overflow-hidden border-4 border-white/30 shadow-2xl backdrop-blur-sm">
                <img
                  src={userPhoto}
                  alt="Prashant Kr. Yadav - Professional Portrait"
                  className="w-full h-full object-cover"
                  data-testid="hero-photo"
                />
                {/* Professional overlay gradient */}
                <div className="absolute inset-0 bg-gradient-to-t from-blue-900/20 via-transparent to-transparent"></div>
              </div>
              
              {/* Enhanced floating elements */}
              <div className="absolute -top-6 -right-6 w-32 h-32 bg-gradient-to-br from-blue-400/30 to-purple-500/30 rounded-full blur-2xl animate-pulse"></div>
              <div className="absolute -bottom-6 -left-6 w-40 h-40 bg-gradient-to-tr from-cyan-400/20 to-blue-500/20 rounded-full blur-2xl animate-pulse delay-1000"></div>
              <div className="absolute top-10 -left-4 w-16 h-16 bg-white/10 rounded-full blur-xl"></div>
              <div className="absolute -top-2 right-20 w-8 h-8 bg-white/20 rounded-full blur-md"></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
