package request;

import java.util.Comparator;

public class Answer implements Comparable<Answer> {

  private final String site;
  private final String item;
  private final Integer price;

  public Answer(String site, String item, Integer price) {
    this.site = site;
    this.item = item;
    this.price = price;
  }

  public boolean isSuccessful(){
      return price!=null;
  }

  public int getPrice() {
    if (price==null){
      throw new IllegalStateException();
    }
    return price;
  }

  public String getItem() {
    return item;
  }

  public String getSite() {
    return site;
  }

  public static Comparator<Answer> ANSWER_COMPARATOR = Comparator.comparing(Answer::getPrice,Comparator.nullsLast(Comparator.naturalOrder())).thenComparing(Answer::getSite,Comparator.naturalOrder()).thenComparing(Answer::getItem,Comparator.naturalOrder());


  @Override
  public String toString() {
    if (price == null) {
      return item + "@" + site + " : Not found";
    } else {
      return item + "@" + site + " : " + price;
    }
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof Answer)) return false;

    Answer answer = (Answer) o;

    if (!site.equals(answer.site)) return false;
    if (!item.equals(answer.item)) return false;
    return price != null ? price.equals(answer.price) : answer.price == null;
  }

  @Override
  public int hashCode() {
    int result = site.hashCode();
    result = 31 * result + item.hashCode();
    result = 31 * result + (price != null ? price.hashCode() : 0);
    return result;
  }

  @Override
  public int compareTo(Answer o) {
      return ANSWER_COMPARATOR.compare(this,o);
  }
}
