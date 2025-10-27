import { ChevronDown } from "lucide-react";

const WhyWellNest = () => {
  return (
    <section id="why" className="py-20 bg-sage-light">
      <div className="container mx-auto px-4 text-center animate-fade-in-up">
        <h2 className="text-5xl sm:text-6xl font-bold text-foreground mb-6" style={{ fontFamily: "'Playfair Display', serif" }}>
          Why WellNest?
        </h2>
        <p className="text-lg text-muted-foreground mb-10">It can help with</p>
        <div className="animate-bounce-slow">
          <ChevronDown className="h-6 w-6 mx-auto text-foreground" strokeWidth={1.5} />
        </div>
      </div>
    </section>
  );
};

export default WhyWellNest;
