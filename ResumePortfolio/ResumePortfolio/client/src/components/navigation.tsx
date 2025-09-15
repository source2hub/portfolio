import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link, useLocation } from "wouter";
import ResumeDownload from "@/components/resume-download";

export default function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 100);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const [location] = useLocation();

  const handleNavigation = (item: any) => {
    if (item.isRoute) {
      // For route navigation, the Link component will handle it
      setIsMobileMenuOpen(false);
    } else {
      // For scroll-to-section navigation
      const element = document.getElementById(item.id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
        setIsMobileMenuOpen(false);
      }
    }
  };

  const navItems = [
    { id: "home", label: "Home", isRoute: false },
    { id: "about", label: "About", isRoute: false },
    { id: "projects", label: "Projects", isRoute: false },
    { id: "skills", label: "Skills", isRoute: false },
    { id: "contact", label: "Contact", isRoute: false },
    { id: "blog", label: "Blog", isRoute: true, route: "/blog" },
  ];

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        isScrolled ? "bg-card/95 backdrop-blur-md" : "bg-card/80 backdrop-blur-md"
      } border-b border-border`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="font-bold text-xl text-primary" data-testid="logo">
            PK Yadav
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <div className="flex space-x-8">
              {navItems.map((item) => (
                item.isRoute ? (
                  <Link key={item.id} href={item.route}>
                    <button
                      className="text-muted-foreground hover:text-primary transition-colors"
                      data-testid={`nav-link-${item.id}`}
                    >
                      {item.label}
                    </button>
                  </Link>
                ) : (
                  <button
                    key={item.id}
                    onClick={() => handleNavigation(item)}
                    className="text-muted-foreground hover:text-primary transition-colors"
                    data-testid={`nav-link-${item.id}`}
                  >
                    {item.label}
                  </button>
                )
              ))}
            </div>
            <ResumeDownload variant="outline" size="sm" />
          </div>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            data-testid="mobile-menu-toggle"
          >
            {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-border" data-testid="mobile-menu">
            <div className="flex flex-col space-y-3">
              {navItems.map((item) => (
                item.isRoute ? (
                  <Link key={item.id} href={item.route}>
                    <button
                      onClick={() => setIsMobileMenuOpen(false)}
                      className="text-muted-foreground hover:text-primary transition-colors text-left px-2 py-1 w-full"
                      data-testid={`mobile-nav-link-${item.id}`}
                    >
                      {item.label}
                    </button>
                  </Link>
                ) : (
                  <button
                    key={item.id}
                    onClick={() => handleNavigation(item)}
                    className="text-muted-foreground hover:text-primary transition-colors text-left px-2 py-1"
                    data-testid={`mobile-nav-link-${item.id}`}
                  >
                    {item.label}
                  </button>
                )
              ))}
              <div className="px-2 pt-3">
                <ResumeDownload variant="outline" size="sm" className="w-full" />
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
