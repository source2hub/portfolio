import { Mail } from "lucide-react";
import { SiLinkedin } from "react-icons/si";

export default function Footer() {
  return (
    <footer className="bg-card border-t border-border py-8" data-testid="footer">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <p className="text-muted-foreground" data-testid="copyright">
            © 2024 Prashant Kr. Yadav. All rights reserved.
          </p>
          <div className="flex justify-center space-x-4 mt-4" data-testid="footer-social-links">
            <a 
              href="mailto:datascienceai50@gmail.com" 
              className="text-muted-foreground hover:text-primary transition-colors"
              data-testid="footer-link-email"
            >
              <Mail className="h-5 w-5" />
            </a>
            <a 
              href="https://linkedin.com/in/prashant-yadav-a2a868277" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-primary transition-colors"
              data-testid="footer-link-linkedin"
            >
              <SiLinkedin className="text-xl" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
