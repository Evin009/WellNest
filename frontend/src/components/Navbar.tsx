import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { User, Bell, ShoppingBag, Menu, X, ChevronDown } from "lucide-react";
import treeLogo from "@/assets/wellnest-tree.png";

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { label: "Home", href: "#home" },
    { label: "login", href: "#login" },
    { label: "quiz", href: "#quiz" },
    { label: "therapist", href: "#therapist" },
    { label: "Why WellNest?", href: "#why" },
    { label: "Services", href: "#services" },
    { label: "Blog", href: "#blog" },
    { label: "Contact", href: "#contact" },
  ];

  return (
    <nav className="bg-background border-b border-border">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between py-3">
          {/* Logo and Tagline - Left */}
          <div className="flex items-center space-x-2">
            <img src={treeLogo} alt="WellNest Tree" className="h-16 w-16" />
            <div className="flex flex-col items-start">
              <h1 
                className="text-4xl font-bold text-foreground leading-none" 
                style={{ fontFamily: "'Dancing Script', cursive" }}
              >
                WellNest
              </h1>
              <p className="text-xs text-foreground mt-1 italic" style={{ fontFamily: "'Playfair Display', serif" }}>
                Smarter matches. Real connections. Better care.
              </p>
            </div>
          </div>

          {/* Desktop Navigation - Center */}
          <div className="hidden lg:flex items-center justify-center space-x-8 flex-1 mx-8">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="text-sm font-normal text-foreground hover:text-primary transition-colors"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                {link.label}
              </a>
            ))}
          </div>

          {/* Icons - Right */}
          <div className="hidden lg:flex items-center justify-end space-x-1">
            <Button variant="ghost" size="icon" className="rounded-full h-8 w-8">
              <ChevronDown className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="rounded-full h-8 w-8 bg-muted">
              <User className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="rounded-full h-8 w-8">
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="rounded-full h-8 w-8 bg-foreground text-background">
              <ShoppingBag className="h-4 w-4" />
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="lg:hidden py-4 bg-background border-t">
            <div className="flex flex-col space-y-3">
              {navLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  className="text-sm font-medium text-foreground hover:text-primary transition-colors px-4 py-2"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
