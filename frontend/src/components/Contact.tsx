import { Phone, Mail, MapPin } from "lucide-react";

const Contact = () => {
  return (
    <section id="contact" className="py-20 bg-background">
      <div className="container mx-auto px-4 text-center animate-fade-in-up">
        <h2 className="text-5xl sm:text-6xl font-bold text-foreground mb-12" style={{ fontFamily: "'Playfair Display', serif" }}>
          Contact Us
        </h2>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-3 text-foreground">
          <span className="text-base">(123)-456-789</span>
          <span className="hidden sm:inline text-base">|</span>
          <a
            href="mailto:contact@wellnest.com"
            className="text-base hover:text-primary transition-colors"
          >
            contact@wellnest.com
          </a>
          <span className="hidden sm:inline text-base">|</span>
          <span className="text-base">Tampa, FL</span>
        </div>
      </div>
    </section>
  );
};

export default Contact;
